# Pokemon Showdown BattleSense - Design Specification

**Project**: Pokemon Showdown AI Battle Assistant
**Team**: Jason Fu (jlf344), Jedi Lertviwatkul (il287), Sun Kongsonthana (nk577)
**Date**: September 11, 2026
**Approach**: Vertical slices - iterative MVP development

## Overview

Build an AI assistant for Pokemon Showdown that predicts opponent actions during Gen 9 OU battles. The system uses machine learning to analyze visible game state and predict the opponent's next move or switch with probability distributions. A browser extension displays these predictions to help players make strategic decisions.

## System Architecture

The system consists of four main components connected via a simple HTTP API:

### 1. Data Pipeline (`src/data/`)

**Parser** (`parse_logs.py`)
- Downloads Gen 8 OU replays from Hugging Face dataset `milkkarten/pokemon-showdown-replays-merged`
- Filters battles with player ratings between 1200-1800
- Samples 50,000-100,000 battles randomly
- Parses Pokemon Showdown log format into structured game states
- Extracts turn-by-turn training examples (game state → opponent action pairs)

**Feature Extractor** (`extract_features.py`)
- Converts raw game state into 56 numerical/categorical features
- Handles partial information (unknown moves, abilities, items for unrevealed Pokemon)
- Outputs feature vectors suitable for XGBoost training

**Dataset Builder** (`create_splits.py`)
- Splits data by battle ID (prevents data leakage)
- 70% training, 15% validation, 15% test
- Handles class imbalance between moves (~80%) and switches (~20%)

### 2. ML Core (`src/models/`)

**Two-stage XGBoost architecture:**

**Stage 1: Action Type Classifier**
- Binary classification: predict "move" vs "switch"
- Input: 56 game state features
- Output: P(move) and P(switch)

**Stage 2a: Move Predictor**
- Multi-class classification: predict which of 4 moves
- Trained only on examples where action type = "move"
- Input: 56 features + 4 move type features
- Output: probabilities over 4 move slots

**Stage 2b: Switch Predictor**
- Multi-class classification: predict which Pokemon to switch to
- Trained only on examples where action type = "switch"
- Input: 56 features + team composition features
- Output: probabilities over available Pokemon (1-5 classes, variable)

**Model Manager** (`model_manager.py`)
- Loads all three trained models
- Combines predictions: P(final) = P(action_type) × P(specific_action)
- Returns top-k predictions sorted by probability

### 3. Evaluation (`src/evaluation/`)

**Metrics Calculator** (`metrics.py`)
- Top-1 accuracy (primary metric)
- Top-3 accuracy (for browser extension display)
- Confidence calibration
- Per-turn analysis (early vs late game)
- Per-situation analysis (winning/losing, type matchups)

**Error Analyzer** (`error_analysis.py`)
- Samples incorrect predictions
- Identifies failure patterns
- Outputs analysis reports for iteration

### 4. Browser Extension (`extension/`)

**Prediction API Server** (`src/api/server.py`)
- Flask server running on localhost:5000
- Loads trained models at startup
- `POST /predict` endpoint accepts game state JSON
- Returns top-3 predictions with probabilities and reasoning

**Extension Components:**
- `content.js`: Reads game state from Pokemon Showdown DOM
- `background.js`: Handles API communication
- `overlay.js`: Displays prediction overlay on battle screen
- `styles.css`: Overlay styling

**Data flow**: Pokemon Showdown → Extension reads DOM → POST to API → Models predict → Extension displays overlay

## Data Pipeline Details

### Data Source

**Hugging Face Dataset**: `milkkarten/pokemon-showdown-replays-merged`
- Format filter: `[Gen 9] OU`
- Rating filter: 1200 ≤ rating ≤ 1800
- Sample size: 50,000-100,000 battles
- Download format: Parquet files using `datasets` library

### Log Parsing

Pokemon Showdown logs are text-based protocol messages:

```
|turn|5
|switch|p2a: Garchomp|Garchomp, L76, M|100/100
|move|p1a: Heatran|Lava Plume|p2a: Garchomp
|-damage|p2a: Garchomp|65/100
|-status|p2a: Garchomp|brn
```

**Parser responsibilities:**
- Track turn boundaries
- Maintain game state (active Pokemon, HP, status, boosts)
- Record team revelation (Gen 9 OU has team preview)
- Identify player actions (moves vs switches)
- Handle edge cases (faints, forced switches, disabled moves)

**Output format:**
```python
{
  "battle_id": "gen9ou-123456789",
  "turn_number": 5,
  "features": [...],  # 56-element feature vector
  "label_action_type": 0,  # 0=move, 1=switch
  "label_move": 2,  # move slot index (0-3) or None
  "label_switch": None  # Pokemon index or None
}
```

### Feature Engineering

**Total: 56 features for MVP**

**Active Pokemon Features (24 features, 12 per side):**
- Species encoding (categorical or type-based grouping)
- Current HP percentage (0.0-1.0)
- Status condition (none/burn/paralyze/sleep/freeze/poison/toxic)
- Stat boost stages for 5 stats: attack, defense, sp.attack, sp.defense, speed (-6 to +6)

**Field Conditions (8 features):**
- Weather (none/sun/rain/sand/hail/snow)
- Terrain (none/electric/grassy/psychic/misty)
- Entry hazards per side: stealth rock, spikes (0-3 layers), toxic spikes (0-2), sticky web

**Team Context (12 features, 6 per side):**
- Number of Pokemon remaining (1-6)
- Number of Pokemon at full HP
- Average HP percentage across team

**Type Matchup (4 features):**
- Offensive type advantage (player → opponent, -2 to +2 scale)
- Defensive type advantage (opponent → player, -2 to +2 scale)
- Speed comparison (based on species + boosts)

**Team Preview Features (8 features):**
- Opponent unrevealed Pokemon count
- For each opponent Pokemon: sent_out (binary), known_moves_count (0-4)

**Handling Unknown Information:**
- Use sentinel value -1 for unknown numeric features
- Add "unknown" category for categorical features
- Model learns to predict based on meta knowledge and partial info

### Train/Val/Test Split

**Critical requirement**: Group by battle_id to prevent data leakage

```python
# Shuffle battles, then split
battles = shuffle(all_battles, seed=42)
train_battles = battles[:35000]  # 70%
val_battles = battles[35000:42500]  # 15%
test_battles = battles[42500:50000]  # 15%

# Extract all turns from battles in each split
train_examples = extract_turns(train_battles)
val_examples = extract_turns(val_battles)
test_examples = extract_turns(test_battles)
```

All turns from a single battle must be entirely in train, val, OR test (never split).

## Machine Learning Models

### Stage 1: Action Type Classifier

**Model**: XGBoost binary classifier

**Hyperparameters** (to tune via validation):
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3
- `n_estimators`: 100-500
- `min_child_weight`: 1-5
- `subsample`: 0.7-1.0
- `colsample_bytree`: 0.7-1.0

**Training**:
- Use all training examples
- Expected class distribution: ~80% moves, ~20% switches
- Handle imbalance via `scale_pos_weight` parameter

**Evaluation**:
- Binary accuracy on validation set
- Precision/recall for both classes
- Confusion matrix

### Stage 2a: Move Selection Classifier

**Model**: XGBoost multi-class classifier (4 classes)

**Input features**:
- Base 56 features
- Plus 4 additional: type of each move (categorical)

**Training**:
- Filter training examples to only those where action_type = "move"
- Multi-class classification over 4 move slots
- Use `multi:softprob` objective

**Challenge**: Move slots are position-dependent but semantically different across battles. Model learns move types + context, not absolute slot positions.

### Stage 2b: Switch Selection Classifier

**Model**: XGBoost multi-class classifier (variable classes)

**Input features**:
- Base 56 features
- Team composition features (species of available Pokemon)

**Training**:
- Filter training examples to only those where action_type = "switch"
- Variable number of classes (1-5 depending on alive Pokemon)
- Handle by encoding as 5-class problem, mask invalid outputs at inference

**Challenge**: Action space varies by game state. At inference, zero out probabilities for fainted Pokemon.

### Combining Predictions

**Inference process:**

1. Run Stage 1 → get P(move) and P(switch)

2. If P(move) > P(switch):
   - Run Stage 2a → get P(move_i) for each move i
   - Final probabilities: P(move_i) = P(move) × P(move_i | move)

3. If P(switch) > P(move):
   - Run Stage 2b → get P(pokemon_j) for each available Pokemon j
   - Mask fainted Pokemon (set prob = 0)
   - Renormalize over alive Pokemon
   - Final probabilities: P(switch_j) = P(switch) × P(pokemon_j | switch)

4. Combine all action probabilities and return top-k

**Example output:**
```python
[
  ("switch", "Toxapex", 0.42),
  ("move", "Earthquake", 0.31),
  ("switch", "Landorus-Therian", 0.15)
]
```

### Model Persistence

Save trained models with versioning:
```
models/
├── v1_action_type_model.json
├── v1_move_model.json
├── v1_switch_model.json
├── v1_preprocessor.pkl
└── model_metadata.json  # hyperparameters, training date, metrics
```

XGBoost models saved as JSON (portable, human-readable).
Preprocessor (feature scaler/encoder) saved as pickle.

## Evaluation Methodology

### Primary Metrics

**1. Top-1 Accuracy**
- Percentage where #1 prediction matches actual action
- Measured separately:
  - Action type accuracy (move vs switch)
  - Move accuracy (given correct action type)
  - Switch accuracy (given correct action type)
  - **Overall accuracy** (combined both stages)

**2. Top-3 Accuracy**
- Percentage where actual action is in top-3 predictions
- Most relevant for browser extension (shows 3 predictions)
- Target: >60% for useful assistance

**3. Confidence Calibration**
- Average probability assigned to correct prediction
- Reliability diagrams (predicted prob vs empirical accuracy)

### Secondary Metrics

**4. Per-Turn Analysis**
- Accuracy by turn number bins (1-10, 11-20, 21-30, 31+)
- Hypothesis: Later turns have better accuracy (more info revealed)

**5. Per-Situation Analysis**
- Accuracy when player is winning vs losing (HP differential)
- Accuracy by type matchup (advantage/neutral/disadvantage)
- Accuracy by opponent Pokemon species (some more predictable)

**6. Confusion Analysis**
- Confusion matrix for action types
- Which moves get confused with each other?
- Which Pokemon switches are most confused?

### Evaluation Process

**Validation Set Usage** (iterative):
- Run predictions on 15% validation set
- Calculate all metrics
- Analyze errors and failure patterns
- Use insights to improve features or tune hyperparameters
- Repeat as needed during development

**Test Set Usage** (once only):
- Final evaluation on 15% test set
- Run only ONCE at end of project
- This is the unbiased performance estimate for the report

### Error Analysis

Build tool (`error_analysis.py`) that:
- Samples 100-200 incorrect predictions from validation set
- For each error, outputs:
  - Turn number and game context
  - Feature vector (human-readable)
  - Model predictions with confidences
  - Actual action
  - Potential explanation (type matchup, HP threshold, meta pattern)

Identify patterns:
- "Model predicts switch when HP < 30%, but aggressive players stay in"
- "Model struggles with uncommon Pokemon (< 1% usage rate)"
- "Move type predictions good, but specific move within type confused"

### Baseline Comparisons

Compare XGBoost model to simple baselines:

**Random Baseline**:
- Uniform random over legal actions
- Expected accuracy: ~10-20% (depending on action space size)

**Type Matchup Baseline**:
- If opponent has type advantage → predict switch
- Else predict most common move for that Pokemon species

**Meta Baseline**:
- Predict most common action for this Pokemon in training data
- No context, pure frequency-based

**HP Threshold Baseline**:
- If HP < 30% → predict switch
- Else predict random move

XGBoost should significantly outperform all baselines.

### Minimum Viable Performance

For MVP to provide value:
- **Top-1 accuracy**: >40% (4x better than random)
- **Top-3 accuracy**: >60% (user sees correct action most of the time)
- **Action type accuracy**: >75% (correctly predicts move/switch)

If these targets aren't met, iterate on features or try different models.

## Browser Extension & API Design

### Architecture

```
Pokemon Showdown (DOM)
    ↓ (JavaScript reads state)
Browser Extension (content.js)
    ↓ (HTTP POST)
Local Flask Server (localhost:5000)
    ↓ (feature extraction + prediction)
XGBoost Models
    ↓ (top-3 predictions)
Browser Extension (overlay.js)
    ↓ (display to user)
Pokemon Showdown UI
```

### Prediction API Server

**Technology**: Flask (lightweight Python web framework)

**File**: `src/api/server.py`

**Startup**:
```python
from flask import Flask, request, jsonify
import xgboost as xgb
import pickle

# Load models once at startup
action_type_model = xgb.Booster()
action_type_model.load_model('models/action_type_model.json')
move_model = xgb.Booster()
move_model.load_model('models/move_model.json')
switch_model = xgb.Booster()
switch_model.load_model('models/switch_model.json')

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Implementation below
    pass

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
```

**Endpoint**: `POST /predict`

**Request format**:
```json
{
  "active_player": {
    "species": "Heatran",
    "hp_percent": 0.72,
    "status": "none",
    "boosts": {"atk": 0, "def": 0, "spa": 1, "spd": 0, "spe": 0},
    "moves": [
      {"name": "Lava Plume", "type": "Fire"},
      {"name": "Flash Cannon", "type": "Steel"},
      {"name": "Earth Power", "type": "Ground"},
      {"name": "Stealth Rock", "type": "Rock"}
    ]
  },
  "active_opponent": {
    "species": "Garchomp",
    "hp_percent": 0.85,
    "status": "none",
    "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
  },
  "opponent_team": [
    {"species": "Garchomp", "alive": true, "active": true},
    {"species": "Toxapex", "alive": true, "active": false},
    {"species": "Landorus-Therian", "alive": true, "active": false},
    {"species": "Zapdos", "alive": true, "active": false},
    {"species": "Weavile", "alive": false, "active": false},
    {"species": "Corviknight", "alive": true, "active": false}
  ],
  "field": {
    "weather": "none",
    "terrain": "none",
    "hazards_player": ["Stealth Rock"],
    "hazards_opponent": []
  }
}
```

**Response format**:
```json
{
  "predictions": [
    {
      "rank": 1,
      "action_type": "switch",
      "target": "Toxapex",
      "probability": 0.42
    },
    {
      "rank": 2,
      "action_type": "move",
      "target": "Earthquake",
      "probability": 0.31
    },
    {
      "rank": 3,
      "action_type": "switch",
      "target": "Landorus-Therian",
      "probability": 0.15
    }
  ],
  "confidence": "medium"
}
```

**Processing logic**:
1. Parse request JSON
2. Convert to 56-element feature vector using preprocessor
3. Run two-stage model prediction
4. Combine probabilities
5. Return top-3 sorted by probability

**Error handling**:
- Invalid JSON → 400 Bad Request
- Missing required fields → 400 Bad Request
- Model prediction error → 500 Internal Server Error
- Log all errors for debugging

### Browser Extension

**Technology**: Vanilla JavaScript (Chrome/Firefox compatible)

**File structure**:
```
extension/
├── manifest.json          # Extension metadata and permissions
├── content.js            # Injected into pokemonshowdown.com
├── background.js         # Service worker for API calls
├── overlay.js            # Prediction display logic
├── styles.css            # Overlay styling
└── icons/                # Extension icons
```

**manifest.json**:
```json
{
  "manifest_version": 3,
  "name": "Pokemon Showdown BattleSense",
  "version": "1.0",
  "description": "AI-powered opponent action predictions",
  "permissions": ["storage"],
  "host_permissions": ["*://play.pokemonshowdown.com/*"],
  "content_scripts": [
    {
      "matches": ["*://play.pokemonshowdown.com/*"],
      "js": ["content.js", "overlay.js"],
      "css": ["styles.css"]
    }
  ],
  "background": {
    "service_worker": "background.js"
  }
}
```

### State Reader (content.js)

**Responsibilities**:
- Detect when it's the player's turn (move selection menu appears)
- Extract game state from Pokemon Showdown DOM
- Format state as JSON matching API schema
- Send to background script for prediction

**DOM extraction strategy**:

Pokemon Showdown uses consistent CSS classes:

**Active Pokemon**:
```javascript
// Player's active Pokemon
const playerPokemon = document.querySelector('.my .pokemon.active');
const playerSpecies = playerPokemon.querySelector('.pokemon-name').textContent;
const playerHPBar = playerPokemon.querySelector('.hpbar');
const playerHP = parseFloat(playerHPBar.dataset.hp) / 100;

// Opponent's active Pokemon
const oppPokemon = document.querySelector('.foe .pokemon.active');
// Similar extraction
```

**Status conditions**:
```javascript
const statusIcon = pokemon.querySelector('.status');
const status = statusIcon ? statusIcon.className.split(' ')[1] : 'none';
```

**Stat boosts**:
Pokemon Showdown displays boost indicators. Parse from battle log:
```javascript
// Look for recent log entries like:
// "Garchomp's Attack rose sharply!" → +2 attack boost
```

**Team composition**:
```javascript
const teamPreview = document.querySelectorAll('.teamicons .picon');
const team = Array.from(teamPreview).map(icon => ({
  species: icon.title.split(',')[0],
  alive: !icon.classList.contains('fainted'),
  active: icon.classList.contains('active')
}));
```

**Field conditions**:
Parse from battle log for weather/terrain messages or check status indicators.

**Timing**:
Run extraction when `.movemenu` appears (player's turn to select action).

### Prediction Display (overlay.js)

**UI Design**:

Create floating overlay positioned right of battle screen:

```
┌─────────────────────────────────┐
│   Opponent Predictions          │
├─────────────────────────────────┤
│ 🔄 Switch to Toxapex       42%  │
│    ██████████████░░░░░░░░       │
│                                 │
│ ⚔️  Use Earthquake         31%  │
│    ██████████░░░░░░░░░░░        │
│                                 │
│ 🔄 Switch to Lando-T       15%  │
│    █████░░░░░░░░░░░░░░░         │
└─────────────────────────────────┘
```

**HTML structure**:
```html
<div id="battlesense-overlay" class="battlesense-container">
  <div class="battlesense-header">Opponent Predictions</div>
  <div class="battlesense-predictions">
    <!-- Populated dynamically -->
  </div>
</div>
```

**CSS** (`styles.css`):
```css
.battlesense-container {
  position: fixed;
  top: 100px;
  right: 20px;
  width: 300px;
  background: rgba(0, 0, 0, 0.85);
  border: 2px solid #4a9eff;
  border-radius: 8px;
  padding: 12px;
  color: white;
  font-family: Verdana, sans-serif;
  z-index: 10000;
}

.battlesense-header {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 10px;
  text-align: center;
  border-bottom: 1px solid #4a9eff;
  padding-bottom: 5px;
}

.prediction-item {
  margin: 8px 0;
  padding: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.prediction-move { border-left: 4px solid #ff6b6b; }
.prediction-switch { border-left: 4px solid #51cf66; }

.probability-bar {
  height: 8px;
  background: linear-gradient(to right, #4a9eff, #1864ab);
  border-radius: 4px;
  margin-top: 4px;
}
```

**Dynamic population**:
```javascript
function displayPredictions(predictions) {
  const container = document.querySelector('.battlesense-predictions');
  container.innerHTML = predictions.map(pred => `
    <div class="prediction-item prediction-${pred.action_type}">
      <div class="prediction-text">
        ${pred.action_type === 'move' ? '⚔️' : '🔄'}
        ${pred.action_type === 'move' ? 'Use' : 'Switch to'} ${pred.target}
        <span class="prediction-prob">${(pred.probability * 100).toFixed(0)}%</span>
      </div>
      <div class="probability-bar" style="width: ${pred.probability * 100}%"></div>
    </div>
  `).join('');
}
```

**Error states**:
- API server offline → Display "Predictions Unavailable (Server Offline)"
- Low confidence (all predictions < 20%) → Display "Uncertain - Many Possibilities"
- Parsing error → Display "Unable to Read Battle State"

### Setup & Usage

**For developers/testers**:

1. Install dependencies:
```bash
pip install flask xgboost pandas scikit-learn
```

2. Train models (or use pre-trained)

3. Start API server:
```bash
python src/api/server.py
# Server running on http://localhost:5000
```

4. Load extension in Chrome:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `extension/` folder

5. Navigate to play.pokemonshowdown.com and start a Gen 9 OU battle

6. Predictions appear automatically each turn

**Future deployment** (post-MVP):
- Host API server on cloud (AWS/GCP/Heroku)
- Package extension for Chrome Web Store
- Add settings panel (API URL, confidence threshold)

## Implementation Process

### Phase 1: Minimal End-to-End System

**Goal**: Prove the concept with simplest version

**Steps**:
1. Download 10k Gen 9 OU battles (rating 1200-1800) from Hugging Face
2. Build log parser to extract game states and opponent actions
3. Create basic feature extractor (~20 features: species, HP, types, boosts only)
4. Split data by battle_id (70/15/15)
5. Train Stage 1 XGBoost (action type only)
6. Build evaluation script measuring accuracy on validation set
7. Create inference script: input game state → output move/switch prediction

**Success criteria**:
- End-to-end pipeline works
- Stage 1 accuracy >70%
- Can manually test on example states

**Estimated effort**: ~2 weeks if working full-time, longer for part-time

### Phase 2: Full Two-Stage System

**Goal**: Add specific action predictions and comprehensive features

**Steps**:
1. Expand dataset to 50k battles
2. Enhance features to full 56-feature set (field conditions, team context, matchups)
3. Train Stage 2a (move selection) and Stage 2b (switch selection)
4. Build model combiner for top-k predictions
5. Implement comprehensive evaluation (top-1, top-3, per-turn, confusion matrices)
6. Build error analysis tool
7. Hyperparameter tuning using validation set
8. Iterate on features based on error analysis

**Success criteria**:
- Top-1 accuracy >40%
- Top-3 accuracy >60%
- Clear understanding of failure modes

**Estimated effort**: ~3-4 weeks

### Phase 3: Browser Extension

**Goal**: Build user-facing interface

**Steps**:
1. Build Flask API server with `/predict` endpoint
2. Create Chrome extension structure
3. Implement state reader (extract game state from DOM)
4. Implement API client (call localhost:5000)
5. Build prediction overlay UI
6. Style overlay (non-intrusive, readable)
7. Test on live battles
8. Handle errors (server offline, parsing failures)
9. Record demo video

**Success criteria**:
- Extension reads state correctly
- Predictions display within 1 second
- No UI conflicts with Pokemon Showdown
- Works across different battle scenarios

**Estimated effort**: ~2-3 weeks

### Phase 4: Refinement & Research

**Goal**: Improve performance and conduct research analysis

**Possible improvements** (prioritize based on Phase 2 results):

**Accuracy**:
- Add temporal features (previous 2-3 turns of history)
- Try neural network models (MLP, LSTM)
- Advanced features (damage calculations, speed tiers)
- Expand to 100k-200k training battles
- Ensemble XGBoost + neural network

**Research**:
- Feature importance analysis (SHAP values)
- Predictability analysis (which Pokemon/matchups easiest?)
- Rating-based comparison (1200-1400 vs 1400-1600 vs 1600-1800)
- Turn-by-turn accuracy curves
- Meta analysis (prediction accuracy vs Pokemon usage rates)

**UX** (optional):
- Add player action recommendations
- Settings panel
- Better styling
- Deploy API server publicly
- Publish to Chrome Web Store

**Estimated effort**: ~4-6 weeks (flexible based on semester timeline)

### Phase 5: Final Evaluation & Report

**Steps**:
1. Run final test set evaluation (ONCE)
2. Compare all model variants tried
3. Generate comprehensive results:
   - Accuracy tables across all metrics
   - Confusion matrices and error analysis
   - Feature importance plots
   - Example predictions (correct and incorrect)
   - Comparison to baselines
4. Write final report:
   - Introduction and motivation
   - Related work (cite papers from proposal)
   - Data collection and preprocessing
   - Feature engineering methodology
   - Model architecture and training
   - Evaluation methodology
   - Results and analysis
   - Discussion of limitations
   - Future work and extensions
5. Create presentation slides
6. Record polished demo video
7. Prepare code repository with documentation

**Deliverables**:
- Final report (PDF, ~10-15 pages)
- Presentation slides
- Demo video (3-5 minutes)
- GitHub repository with README and setup instructions
- Trained models and sample data

**Estimated effort**: ~2 weeks for documentation and presentation prep

## Technical Considerations

### Data Processing

**Libraries**:
- `datasets` (Hugging Face) - efficient Parquet loading
- `pandas` - data manipulation
- `numpy` - numerical operations

**Storage**:
- Raw logs: ~2-5 GB for 50k battles
- Processed features: ~500 MB
- Trained models: ~50-100 MB

**Processing time estimates**:
- Download 50k battles: 10-30 minutes (depending on network)
- Parse logs: 1-2 hours
- Extract features: 30 minutes
- Train models: 10-30 minutes per model (with hyperparameter search: 2-4 hours)

### Model Training

**Hardware requirements**:
- CPU: Modern multi-core (XGBoost is CPU-optimized)
- RAM: 8 GB minimum, 16 GB recommended
- Disk: 10 GB free space
- GPU: Not required for XGBoost

**Training strategy**:
- Use 5-fold cross-validation on training set for hyperparameter search
- Random search over hyperparameter space (50-100 combinations)
- Early stopping based on validation loss (patience=10 rounds)
- Save best model based on validation accuracy

**XGBoost-specific optimizations**:
- Set `n_jobs=-1` (use all CPU cores)
- Use `hist` tree method (faster for large datasets)
- Enable early stopping to avoid overfitting

### Browser Extension

**Compatibility**:
- Primary: Chrome (Manifest V3)
- Secondary: Firefox (mostly compatible)
- Safari: Not planned (requires different approach)

**Permissions needed**:
- `storage` - save user preferences
- `host_permissions` for pokemonshowdown.com - read DOM

**Performance**:
- State extraction: <100ms
- API call roundtrip: <500ms (local server)
- Total latency: <1 second per prediction

**Robustness**:
- Pokemon Showdown DOM structure may change → monitor for breakage
- Fallback gracefully if elements not found
- Log errors to console for debugging

### API Server

**Production considerations** (post-MVP):
- Add request rate limiting (prevent abuse)
- Add authentication (API keys)
- Use CORS properly for cross-origin requests
- Deploy on cloud with autoscaling
- Add logging and monitoring
- Containerize with Docker

**For MVP (local-only)**:
- Simple Flask server on localhost
- No authentication needed
- CORS enabled for browser extension
- Basic error logging

## Success Criteria

### Technical Success

**Phase 1**:
- ✅ Can download and parse Pokemon Showdown logs
- ✅ Stage 1 model trains without errors
- ✅ Accuracy >70% on action type prediction

**Phase 2**:
- ✅ Top-1 accuracy >40%
- ✅ Top-3 accuracy >60%
- ✅ All three models train successfully
- ✅ Error analysis identifies clear failure patterns

**Phase 3**:
- ✅ Extension installs and runs on Pokemon Showdown
- ✅ Predictions display correctly
- ✅ Latency <1 second
- ✅ Demo video shows working system

### Research Success

- ✅ Demonstrated that ML can predict opponent actions better than baselines
- ✅ Identified which features are most important for prediction
- ✅ Analyzed when/why model fails (turn number, matchup type, etc.)
- ✅ Compared two-stage vs single-stage approach (if time permits)
- ✅ Comprehensive final report documenting methodology and results

### User Value

- ✅ Browser extension provides useful information to players
- ✅ Predictions are accurate enough to influence decision-making
- ✅ System works on live battles, not just replay analysis
- ✅ UI is non-intrusive and readable

## Limitations & Future Work

### Known Limitations

**Model limitations**:
- Assumes rational play (struggles with unexpected/suboptimal moves)
- No adaptation to individual opponent style
- Can't predict long-term strategy (only next move)
- Limited by features (doesn't calculate exact damage, speed values)

**Data limitations**:
- Only trained on 1200-1800 rating range
- May not generalize to very high or very low skill levels
- Gen 9 OU meta evolves over time (model may become outdated)

**System limitations**:
- Requires local API server (not user-friendly for non-technical users)
- Extension may break if Pokemon Showdown updates DOM structure
- No mobile support

### Future Extensions

**Technical improvements**:
- **Temporal models**: LSTM/Transformer to use battle history
- **Ensemble methods**: Combine XGBoost + neural network
- **Multi-tier training**: Separate models for different rating ranges
- **Damage calculator integration**: Use exact damage formulas as features
- **Real-time learning**: Update model based on current meta

**Feature additions**:
- **Action recommendations**: Not just predict opponent, but suggest player's best move
- **Probability explanations**: "Likely to switch because type disadvantage"
- **Team analysis**: Predict opponent's team composition from partial reveals
- **Replay analysis**: Upload replay, get analysis of critical decisions

**Deployment**:
- Cloud-hosted API (no local server needed)
- Mobile app (not just browser extension)
- Integration with Pokemon Showdown (if allowed by site)

**Research extensions**:
- **Multi-agent learning**: Model both players simultaneously
- **Reinforcement learning**: Train agent to play optimally
- **Meta-game analysis**: Track how predictability changes with meta shifts
- **Cross-generation transfer**: Does Gen 9 model work on Gen 8?

## Risk Mitigation

### Technical Risks

**Risk**: Log parsing is too complex
- **Mitigation**: Use existing library (poke-env) or start with simple subset of log events

**Risk**: Model accuracy too low to be useful
- **Mitigation**: Start with baselines to set expectations; iterate on features; try multiple model types

**Risk**: Pokemon Showdown DOM changes break extension
- **Mitigation**: Use robust selectors; test on multiple pages; document known structure

**Risk**: Dataset too large to process
- **Mitigation**: Start with 10k battles, scale up as needed; use efficient libraries (Parquet, XGBoost)

### Project Management Risks

**Risk**: Team member unavailability
- **Mitigation**: Use version control (git); document code; modular architecture allows parallel work

**Risk**: Running out of time before deadline
- **Mitigation**: Vertical slice approach ensures working system at each phase; MVP delivers value even without all features

**Risk**: Scope creep
- **Mitigation**: Clearly defined MVP; defer enhancements to Phase 4; prioritize based on research goals

## Appendix: File Structure

```
pokemon-showdown-battlesense/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/                      # Downloaded replay logs
│   ├── processed/                # Parsed training examples
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── metadata.json             # Dataset statistics
├── src/
│   ├── data/
│   │   ├── download_data.py      # Hugging Face download
│   │   ├── parse_logs.py         # Log parser
│   │   ├── extract_features.py   # Feature engineering
│   │   └── create_splits.py      # Train/val/test split
│   ├── models/
│   │   ├── train_stage1.py       # Action type classifier
│   │   ├── train_stage2a.py      # Move predictor
│   │   ├── train_stage2b.py      # Switch predictor
│   │   └── model_manager.py      # Combined inference
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy calculations
│   │   ├── evaluate.py           # Run evaluation
│   │   └── error_analysis.py     # Failure pattern analysis
│   ├── api/
│   │   └── server.py             # Flask API server
│   └── utils/
│       ├── constants.py          # Pokemon types, species lists
│       └── preprocessing.py      # Feature scaling/encoding
├── extension/
│   ├── manifest.json
│   ├── content.js                # State reader
│   ├── background.js             # API communication
│   ├── overlay.js                # Prediction display
│   ├── styles.css
│   └── icons/
├── models/                       # Saved trained models
│   ├── action_type_model.json
│   ├── move_model.json
│   ├── switch_model.json
│   ├── preprocessor.pkl
│   └── metadata.json
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   ├── feature_analysis.ipynb
│   └── model_comparison.ipynb
├── tests/                        # Unit tests
│   ├── test_parser.py
│   ├── test_features.py
│   └── test_models.py
├── results/                      # Evaluation outputs
│   ├── metrics/
│   ├── plots/
│   └── error_samples/
└── docs/
    ├── final_report.pdf
    ├── presentation.pdf
    └── demo_video.mp4
```

## Conclusion

This design provides a complete blueprint for building Pokemon Showdown BattleSense using a vertical slice approach. The system combines data engineering, machine learning, and web development to create a practical tool that demonstrates the predictability of opponent actions in competitive Pokemon battles.

The two-stage XGBoost architecture balances accuracy and interpretability, while the browser extension provides immediate user value. The phased implementation ensures working code at every stage, reducing risk and enabling iterative improvements.

Key design decisions:
- **Two-stage classification** (easier to train and debug than single massive classifier)
- **XGBoost for MVP** (strong baseline, interpretable, fast training)
- **Local API server** (simple deployment for MVP)
- **Simple overlay** (minimal UI, focuses on predictions)
- **Vertical slices** (working system from day one)

Success metrics are clear: >40% top-1 accuracy, >60% top-3 accuracy. Error analysis will guide iterations. The system is extensible for future enhancements (temporal features, neural networks, action recommendations).

This design balances research goals (understanding opponent predictability), technical goals (building accurate ML models), and user value (practical battle assistance).
