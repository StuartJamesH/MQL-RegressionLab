//+------------------------------------------------------------------+
//|                                              TransformerTrader.mq5 |
//| Expert Advisor using the Causal Patch Transformer for trading    |
//|                                                                  |
//| Uses the ONNX model exported by DeploymentPackager to generate   |
//| trade signals. Implements:                                       |
//|   - Signal generation via DistributionalSignalGenerator logic    |
//|   - Kelly-based position sizing                                  |
//|   - Risk management (trailing stops, exposure limits)            |
//|   - Walk-forward compatible (no lookahead)                       |
//|                                                                  |
//| Dependencies:                                                    |
//|   - TransformerModel indicator (provides signal buffer)          |
//|   - ONNX runtime (MQL5 built-in)                                 |
//|   - model.onnx in Common/Files/                                  |
//+------------------------------------------------------------------+
#property copyright "MQL-RegressionLab"
#property link      ""
#property version   "1.00"

// --- Input Parameters ---

// Model
input string InpModelPath        = "model.onnx";  // ONNX model file
input int    InpSeqLen           = 256;            // Sequence length
input int    InpNFeatures        = 32;             // Number of features
input int    InpPrimaryHorizon   = 2;              // Primary horizon index (0-based)

// Trade Management
input double InpMaxRiskPerTrade  = 0.02;           // Max risk per trade (fraction of equity)
input double InpTPMultiplier     = 3.0;            // Take-profit ATR multiplier
input double InpSLMultiplier     = 1.5;            // Stop-loss ATR multiplier
input int    InpMaxHoldBars      = 120;            // Maximum hold duration (bars)
input double InpSignalThreshold  = 0.1;            // Minimum |signal| to trade

// Risk Management
input int    InpMaxPositions     = 3;              // Max concurrent positions
input double InpMaxExposurePct   = 0.15;           // Max total exposure
input double InpTrailingAtrMult  = 1.5;            // Trailing stop ATR multiplier
input bool   InpHalfKelly        = true;           // Use half-Kelly sizing

// Position Sizing
input double InpMaxPositionPct   = 0.05;           // Max position as fraction of equity
input double InpAccountRiskPct   = 0.02;           // Account risk per trade (for Kelly)

// --- Global Variables ---
long    g_onnx_session = INVALID_HANDLE;
double  g_norm_mean[];       // Feature normalizer means
double  g_norm_std[];        // Feature normalizer stds
int     g_n_horizons = 5;    // Number of forecast horizons
int     g_n_regimes  = 4;    // Number of regime classes

// Position tracking
struct TradePosition {
   ulong    ticket;
   int      direction;       // 1=long, -1=short
   double   entry_price;
   double   current_sl;
   double   current_tp;
   double   trailing_sl;
   double   entry_atr;
   datetime entry_time;
   double   position_size;
   bool     active;
};
TradePosition g_positions[];

//+------------------------------------------------------------------+
//| Expert Initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
    // --- Load ONNX model ---
    // PLACEHOLDER: In production, load via OnnxCreate()
    //
    // string full_path = InpModelPath;
    // g_onnx_session = OnnxCreate(full_path, ONNX_DEBUG_MODE);
    // if(g_onnx_session == INVALID_HANDLE) {
    //     Print("Failed to load ONNX model: ", full_path);
    //     return INIT_FAILED;
    // }
    // Print("ONNX model loaded: ", full_path);

    // --- Load normalizer stats from normalizer.json ---
    // PLACEHOLDER: In production, read from Common/Files/normalizer.json
    //
    // For now, use identity normalizer (mean=0, std=1).
    ArrayResize(g_norm_mean, InpNFeatures);
    ArrayResize(g_norm_std,  InpNFeatures);
    ArrayInitialize(g_norm_mean, 0.0);
    ArrayInitialize(g_norm_std,  1.0);

    // --- Initialize position array ---
    ArrayResize(g_positions, 0);

    Print("TransformerTrader EA initialized.");
    Print("  SeqLen=", InpSeqLen, " Features=", InpNFeatures,
          " Horizons=", g_n_horizons, " Regimes=", g_n_regimes);
    Print("  Risk per trade: ", InpMaxRiskPerTrade * 100, "%");
    Print("  Max positions: ",  InpMaxPositions);
    Print("  Signal threshold: ", InpSignalThreshold);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // --- Close all positions ---
    for(int i = 0; i < ArraySize(g_positions); i++) {
        if(g_positions[i].active) {
            // ClosePosition(g_positions[i].ticket);
        }
    }

    // --- Release ONNX session ---
    // if(g_onnx_session != INVALID_HANDLE) {
    //     OnnxRelease(g_onnx_session);
    // }

    Print("TransformerTrader EA removed.");
}

//+------------------------------------------------------------------+
//| OnTick — Main trading logic                                      |
//+------------------------------------------------------------------+
void OnTick()
{
    // --- 1. Generate signal from model ---
    double signal = GetModelSignal();

    // --- 2. Update existing positions (trailing stops) ---
    UpdateTrailingStops();

    // --- 3. Check if we should enter a new position ---
    if(MathAbs(signal) >= InpSignalThreshold &&
       CountActivePositions() < InpMaxPositions)
    {
        int direction = (signal > 0) ? 1 : -1;

        // Compute position size using Kelly
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double current_exposure = GetTotalExposure();
        double size = ComputePositionSize(equity, current_exposure);

        if(size > 0.0) {
            OpenPosition(direction, size);
        }
    }

    // --- 4. Check exit conditions for open positions ---
    CheckExits(signal);
}

//+------------------------------------------------------------------+
//| GetModelSignal — Run ONNX inference and return signal [-1, 1]    |
//+------------------------------------------------------------------+
double GetModelSignal()
{
    // PLACEHOLDER: Full ONNX inference pipeline
    //
    // Steps:
    //   1. Compute features for the last InpSeqLen bars
    //      using the feature specification from FeatureSpec.
    //   2. Normalize features using g_norm_mean/g_norm_std.
    //   3. Run ONNX inference via OnnxRun().
    //   4. Parse outputs to compute signal:
    //      a. mu = output[primary_horizon]
    //      b. sigma = exp(output[n_horizons + primary_horizon])
    //      c. direction_logits = output[2*n_horizons + ...]
    //      d. dir_up = softmax(direction_logits)[UP]
    //      e. dir_down = softmax(direction_logits)[DOWN]
    //      f. sharpe = mu / (sigma + 1e-6)
    //      g. dir_signal = dir_up - dir_down
    //      h. signal = tanh(sharpe * dir_signal)
    //      i. Regime gate: if extreme regime, signal = 0
    //      j. Threshold: if |signal| < threshold, signal = 0
    //
    // Returns: signal value in [-1, 1]

    // Placeholder: return 0 (no signal)
    return 0.0;
}

//+------------------------------------------------------------------+
//| ComputePositionSize — Kelly-based position sizing                |
//+------------------------------------------------------------------+
double ComputePositionSize(double equity, double current_exposure)
{
    // PLACEHOLDER: Kelly criterion position sizing
    //
    // f* = (p_win * avg_win - p_loss * avg_loss) / (avg_win * avg_loss)
    // half-Kelly: f = f* / 2
    // size = min(f * equity, max_position_pct * equity - current_exposure)
    //
    // Parameters:
    //   p_win  = estimated win probability from direction head
    //   avg_win = average TP reward (in account %)
    //   avg_loss = average SL risk (in account %)

    if(equity <= 0.0) return 0.0;

    // Placeholder values
    double p_win    = 0.45;
    double avg_win  = InpTPMultiplier * InpAccountRiskPct;
    double avg_loss = InpSLMultiplier * InpAccountRiskPct;

    double f_star = (p_win * avg_win - (1.0 - p_win) * avg_loss)
                  / (avg_win * avg_loss);

    if(InpHalfKelly) f_star *= 0.5;
    if(f_star < 0.0) return 0.0;

    f_star = MathMin(f_star, InpMaxPositionPct);

    double remaining = InpMaxPositionPct * equity - current_exposure;
    double size = MathMin(f_star * equity, remaining);

    return MathMax(size, 0.0);
}

//+------------------------------------------------------------------+
//| OpenPosition — Execute a new trade                                |
//+------------------------------------------------------------------+
void OpenPosition(int direction, double size)
{
    // PLACEHOLDER: Position opening logic
    //
    // 1. Compute entry price (current Ask for long, Bid for short)
    // 2. Compute SL/TP levels based on ATR
    //    - tp = entry +/- InpTPMultiplier * ATR
    //    - sl = entry -/+ InpSLMultiplier * ATR
    // 3. Send order via OrderSend()
    // 4. Track position in g_positions[]

    Print("Opening ", (direction > 0 ? "LONG" : "SHORT"),
          " position, size=", DoubleToString(size, 2));
}

//+------------------------------------------------------------------+
//| CheckExits — Evaluate exit conditions                             |
//+------------------------------------------------------------------+
void CheckExits(double current_signal)
{
    // PLACEHOLDER: Exit logic
    //
    // For each active position:
    //   1. Check if TP or SL hit (intra-bar using High/Low)
    //   2. Check max hold duration
    //   3. Check signal reversal (if signal flips and is strong)
    //   4. Close position if any condition met

    for(int i = 0; i < ArraySize(g_positions); i++)
    {
        if(!g_positions[i].active) continue;

        // --- Check TP/SL ---
        // double current_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        // double current_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        // if(g_positions[i].direction == 1 && current_bid >= g_positions[i].current_tp)
        //     ClosePosition(g_positions[i].ticket, "TP");
        // else if(g_positions[i].direction == -1 && current_ask <= g_positions[i].current_tp)
        //     ClosePosition(g_positions[i].ticket, "TP");

        // --- Check duration ---
        // if(TimeCurrent() - g_positions[i].entry_time > InpMaxHoldBars * PeriodSeconds())
        //     ClosePosition(g_positions[i].ticket, "Timeout");
    }
}

//+------------------------------------------------------------------+
//| UpdateTrailingStops — Move stops in favorable direction          |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
    // PLACEHOLDER: Trailing stop logic
    //
    // For each active position:
    //   Long:  new_sl = max(current_sl, recent_high - InpTrailingAtrMult * ATR)
    //   Short: new_sl = min(current_sl, recent_low  + InpTrailingAtrMult * ATR)
    //   Only update if new_sl is more favorable.
    //   If new_sl differs, modify the order via OrderModify().

    for(int i = 0; i < ArraySize(g_positions); i++)
    {
        if(!g_positions[i].active) continue;

        // double atr = iATR(_Symbol, PERIOD_CURRENT, 14, 0);
        // double recent_high = iHigh(_Symbol, PERIOD_CURRENT, 1);
        // double recent_low  = iLow(_Symbol, PERIOD_CURRENT, 1);
        //
        // if(g_positions[i].direction == 1) {
        //     double new_sl = recent_high - InpTrailingAtrMult * atr;
        //     if(new_sl > g_positions[i].trailing_sl) {
        //         g_positions[i].trailing_sl = new_sl;
        //         // ModifyStopLoss(g_positions[i].ticket, new_sl);
        //     }
        // } else {
        //     double new_sl = recent_low + InpTrailingAtrMult * atr;
        //     if(new_sl < g_positions[i].trailing_sl) {
        //         g_positions[i].trailing_sl = new_sl;
        //         // ModifyStopLoss(g_positions[i].ticket, new_sl);
        //     }
        // }
    }
}

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+

int CountActivePositions()
{
    int count = 0;
    for(int i = 0; i < ArraySize(g_positions); i++) {
        if(g_positions[i].active) count++;
    }
    return count;
}

double GetTotalExposure()
{
    double expo = 0.0;
    for(int i = 0; i < ArraySize(g_positions); i++) {
        if(g_positions[i].active)
            expo += MathAbs(g_positions[i].position_size);
    }
    return expo;
}
//+------------------------------------------------------------------+
