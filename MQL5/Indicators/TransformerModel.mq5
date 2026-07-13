//+------------------------------------------------------------------+
//|                                              TransformerModel.mq5 |
//| MQL5 ONNX Inference Wrapper for the Causal Patch Transformer     |
//|                                                                  |
//| Loads the ONNX model exported by Python's DeploymentPackager     |
//| and runs inference on a sliding window of bar features.          |
//|                                                                  |
//| Requirements:                                                    |
//|   - ONNX Runtime in MQL5 (OnnxRuntime.mqh)                       |
//|   - model.onnx in Common/Files/                                  |
//|   - Feature computation matching Python's FeatureSpec            |
//|                                                                  |
//| Output:                                                          |
//|   - 6 indicator buffers: mu, sigma, dir_up, dir_flat, dir_down, |
//|     regime, volatility — one per prediction head                 |
//+------------------------------------------------------------------+
#property copyright "MQL-RegressionLab"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 8
#property indicator_plots   1

// --- Input Parameters ---
input int    InpSeqLen       = 256;   // Input sequence length (bars)
input int    InpNFeatures    = 32;    // Number of feature columns
input int    InpHorizonIdx   = 2;     // Which horizon to display (0-based)
input string InpModelPath    = "model.onnx";  // ONNX model file in Common/Files/

// --- ONNX Input/Output names (must match Python export) ---
#define ONNX_INPUT_NAME        "input"
#define ONNX_OUTPUT_MU         "mu"
#define ONNX_OUTPUT_SIGMA      "log_sigma"
#define ONNX_OUTPUT_DIRECTION  "direction_logits"
#define ONNX_OUTPUT_REGIME     "regime_logits"
#define ONNX_OUTPUT_VOLATILITY "volatility"
#define ONNX_OUTPUT_QUANTILES  "quantiles"

// --- Display Buffer ---
double SignalBuffer[];        // Primary signal (mu at selected horizon)

// --- Calculation Buffers (not displayed) ---
double MuBuffer[];            // All horizons' mu predictions
double SigmaBuffer[];         // All horizons' sigma
double DirUpBuffer[];         // P(up) at selected horizon
double DirDownBuffer[];       // P(down) at selected horizon
double VolBuffer[];           // Volatility at selected horizon
double RegimeBuffer[];        // Regime class prediction
double FeatureMatrix[];       // Raw feature buffer for ONNX input (flat)

// --- ONNX Session Handle ---
long g_onnx_session = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Indicator Initialization                                         |
//+------------------------------------------------------------------+
int OnInit()
{
    // --- Bind buffers ---
    SetIndexBuffer(0, SignalBuffer,     INDICATOR_DATA);
    SetIndexBuffer(1, MuBuffer,         INDICATOR_CALCULATIONS);
    SetIndexBuffer(2, SigmaBuffer,      INDICATOR_CALCULATIONS);
    SetIndexBuffer(3, DirUpBuffer,      INDICATOR_CALCULATIONS);
    SetIndexBuffer(4, DirDownBuffer,    INDICATOR_CALCULATIONS);
    SetIndexBuffer(5, VolBuffer,        INDICATOR_CALCULATIONS);
    SetIndexBuffer(6, RegimeBuffer,     INDICATOR_CALCULATIONS);
    SetIndexBuffer(7, FeatureMatrix,    INDICATOR_CALCULATIONS);

    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
    PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpSeqLen);
    PlotIndexSetString(0, PLOT_LABEL, "Transformer Signal");

    IndicatorSetString(INDICATOR_SHORTNAME,
                       StringFormat("TransformerModel(%d,%d)", InpSeqLen, InpNFeatures));

    // --- Load ONNX model ---
    // NOTE: In production, load the ONNX model from Common/Files/
    // using OnnxCreate() or the built-in ONNX API.
    //
    // Example (pseudo-code):
    //   string model_path = InpModelPath;
    //   g_onnx_session = OnnxCreate(model_path, ONNX_DEBUG_MODE);
    //   if(g_onnx_session == INVALID_HANDLE) {
    //       Print("Failed to load ONNX model: ", model_path);
    //       return INIT_FAILED;
    //   }
    //
    // PLACEHOLDER: Model loading will be implemented when ONNX runtime
    // is available in the MQL5 build environment.

    Print("TransformerModel initialized (ONNX loading is PLACEHOLDER).");
    Print("  Seq length: ", InpSeqLen, ", Features: ", InpNFeatures);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Indicator Deinitialization                                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // --- Release ONNX session ---
    // if(g_onnx_session != INVALID_HANDLE) {
    //     OnnxRelease(g_onnx_session);
    //     g_onnx_session = INVALID_HANDLE;
    // }
}

//+------------------------------------------------------------------+
//| ComputeFeatures — Builds the feature vector for bar `idx`.       |
//|                                                                  |
//| Must exactly match the FeatureSpec used during Python training.  |
//| Features are normalized using the mean/std from normalizer.json. |
//+------------------------------------------------------------------+
void ComputeFeatures(
    int               idx,
    const double     &open[],
    const double     &high[],
    const double     &low[],
    const double     &close[],
    const datetime   &time[],
    double           &features[])   // output: array[N_FEATURES]
{
    if(idx < 1) {
        ArrayInitialize(features, 0.0);
        return;
    }

    int f = 0;

    // --- Feature 0: log_return ---
    if(close[idx - 1] > 0.0)
        features[f++] = MathLog(close[idx] / close[idx - 1]);
    else
        features[f++] = 0.0;

    // --- Feature 1-3: OHLC relative to Close ---
    features[f++] = (open[idx]  - close[idx]) / close[idx];    // O_rel
    features[f++] = (high[idx]  - close[idx]) / close[idx];    // H_rel
    features[f++] = (low[idx]   - close[idx]) / close[idx];    // L_rel

    // --- Feature 4-7: Session features (cyclic time) ---
    MqlDateTime dt;
    TimeToStruct(time[idx], dt);
    features[f++] = MathSin(2.0 * M_PI * dt.hour / 24.0);       // hour_sin
    features[f++] = MathCos(2.0 * M_PI * dt.hour / 24.0);       // hour_cos
    features[f++] = MathSin(2.0 * M_PI * dt.day_of_week / 7.0); // dow_sin
    features[f++] = MathCos(2.0 * M_PI * dt.day_of_week / 7.0); // dow_cos

    // --- Feature 8-10: Momentum ---
    if(idx >= 5  && close[idx - 5]  > 0.0)
        features[f++] = close[idx] / close[idx - 5]  - 1.0;    // mom_5
    else features[f++] = 0.0;

    if(idx >= 20 && close[idx - 20] > 0.0)
        features[f++] = close[idx] / close[idx - 20] - 1.0;    // mom_20
    else features[f++] = 0.0;

    if(idx >= 60 && close[idx - 60] > 0.0)
        features[f++] = close[idx] / close[idx - 60] - 1.0;    // mom_60
    else features[f++] = 0.0;

    // --- Feature 11: High-Low range relative to Close ---
    features[f++] = (high[idx] - low[idx]) / close[idx];

    // --- Feature 12: Ret vol scaled ---
    // (rolling std of log returns over 20 bars, then log_return / std)
    if(idx >= 20) {
        double sum = 0.0, sum_sq = 0.0;
        for(int i = idx - 19; i <= idx; i++) {
            double lr = (close[i-1] > 0.0) ? MathLog(close[i] / close[i-1]) : 0.0;
            sum += lr;
            sum_sq += lr * lr;
        }
        double mean_lr = sum / 20.0;
        double std_lr  = MathSqrt(sum_sq / 20.0 - mean_lr * mean_lr);
        if(std_lr > 1e-8)
            features[f++] = features[0] / std_lr;  // log_ret / rolling_std
        else
            features[f++] = 0.0;
    } else {
        features[f++] = 0.0;
    }

    // --- Pad remaining features to N_FEATURES ---
    while(f < InpNFeatures) {
        features[f++] = 0.0;
    }

    // --- Normalize (z-score) ---
    // TODO: Load mean/std from normalizer.json
    // For now, use identity normalization (placeholder).
    // In production:
    //   for(int i = 0; i < InpNFeatures; i++)
    //       features[i] = (features[i] - g_norm_mean[i]) / g_norm_std[i];
}

//+------------------------------------------------------------------+
//| RunONNXInference — Runs the ONNX model on a feature window.      |
//+------------------------------------------------------------------+
void RunONNXInference(
    int               anchor_bar,     // current bar index
    const double     &features_flat[], // flat array of [seq_len * n_features]
    double           &outputs[])       // output buffer (size depends on heads)
{
    // PLACEHOLDER: ONNX inference
    //
    // In production:
    //   // Prepare input tensor (1, seq_len, n_features) as flat float array
    //   // Set input shape: {1, seq_len, n_features}
    //   OnnxSetInputShape(g_onnx_session, ONNX_INPUT_NAME, shape);
    //   OnnxRun(g_onnx_session, ONNX_INPUT_NAME, features_flat, outputs);
    //
    // For now, fill outputs with zeros as placeholder.
    ArrayInitialize(outputs, 0.0);

    Print("ONNX inference called (PLACEHOLDER) at bar ", anchor_bar);
}

//+------------------------------------------------------------------+
//| OnCalculate — Main indicator loop                                 |
//+------------------------------------------------------------------+
int OnCalculate(
    const int       rates_total,
    const int       prev_calculated,
    const datetime &time[],
    const double   &open[],
    const double   &high[],
    const double   &low[],
    const double   &close[],
    const long     &tick_volume[],
    const long     &volume[],
    const int      &spread[])
{
    if(rates_total < InpSeqLen) return 0;

    // Use oldest-first indexing
    ArraySetAsSeries(close, false);
    ArraySetAsSeries(open,  false);
    ArraySetAsSeries(high,  false);
    ArraySetAsSeries(low,   false);
    ArraySetAsSeries(time,  false);

    int start = prev_calculated - 1;
    if(start < InpSeqLen) start = InpSeqLen;

    // --- Feature computation for each new bar ---
    for(int i = start; i < rates_total; i++)
    {
        // --- Build feature window for this bar ---
        // We need the last InpSeqLen bars ending at bar i.
        // Extract features for bars [i - InpSeqLen + 1, i].
        //
        // In production, features are pre-computed and stacked
        // into a 2D array of shape (InpSeqLen, InpNFeatures).

        int window_start = i - InpSeqLen + 1;
        double features_window[];  // flat: [InpSeqLen * InpNFeatures]
        ArrayResize(features_window, InpSeqLen * InpNFeatures);

        for(int w = 0; w < InpSeqLen; w++)
        {
            double bar_features[];
            ArrayResize(bar_features, InpNFeatures);

            int bar_idx = window_start + w;
            ComputeFeatures(bar_idx, open, high, low, close, time, bar_features);

            // Copy into flat buffer
            for(int f = 0; f < InpNFeatures; f++)
                features_window[w * InpNFeatures + f] = bar_features[f];
        }

        // --- Run ONNX inference ---
        // Output size depends on heads:
        //   mu:           n_horizons
        //   log_sigma:    n_horizons
        //   direction:    n_horizons * 3
        //   regime:       n_regime_classes
        //   volatility:   n_horizons
        //   quantiles:    n_horizons * 3
        //
        // Total output size ≈ n_horizons * 8 + n_regime_classes

        int n_horizons = 5;
        int n_regimes  = 4;
        int output_size = n_horizons * 2        // mu + sigma
                        + n_horizons * 3         // direction logits
                        + n_regimes              // regime
                        + n_horizons             // volatility
                        + n_horizons * 3;        // quantiles

        double outputs[];
        ArrayResize(outputs, output_size);

        RunONNXInference(i, features_window, outputs);

        // --- Parse outputs ---
        // Offset tracking (must match Python export order)
        int off = 0;

        // mu: n_horizons floats
        // At selected horizon:
        int h = InpHorizonIdx;
        double mu_val       = outputs[off + h];
        off += n_horizons;

        // log_sigma: n_horizons floats
        double sigma_val    = MathExp(outputs[off + h]) + 1e-6;
        off += n_horizons;

        // direction_logits: n_horizons * 3
        double dir_up       = outputs[off + h * 3 + 2];  // class 2 = UP
        double dir_down     = outputs[off + h * 3 + 0];  // class 0 = DOWN
        off += n_horizons * 3;

        // regime_logits: n_regimes
        double regime_pred  = 0;
        double max_logit    = -1e10;
        for(int r = 0; r < n_regimes; r++) {
            if(outputs[off + r] > max_logit) {
                max_logit = outputs[off + r];
                regime_pred = (double)r;
            }
        }
        off += n_regimes;

        // volatility: n_horizons
        double vol_val = MathExp(outputs[off + h]);  // log_vol -> vol
        off += n_horizons;

        // quantiles: n_horizons * 3 (skip for now)
        // off += n_horizons * 3;

        // --- Compute signal ---
        // Sharpe-like: s = mu / sigma
        double sharpe = (sigma_val > 1e-6) ? mu_val / sigma_val : 0.0;

        // Directional confidence
        double dir_confidence = MathMax(
            MathExp(dir_up) / (MathExp(dir_up) + MathExp(dir_down) + 1.0),
            MathExp(dir_down) / (MathExp(dir_up) + MathExp(dir_down) + 1.0)
        );
        double dir_signal = (MathExp(dir_up) - MathExp(dir_down))
                          / (MathExp(dir_up) + MathExp(dir_down) + 1.0);

        // Composite signal: tanh(sharpe * dir_signal)
        double signal = MathTanh(sharpe * dir_signal);

        // --- Store to buffers ---
        SignalBuffer[i]  = signal;
        MuBuffer[i]      = mu_val;
        SigmaBuffer[i]   = sigma_val;
        DirUpBuffer[i]   = dir_confidence;
        DirDownBuffer[i] = 1.0 - dir_confidence;
        VolBuffer[i]     = vol_val;
        RegimeBuffer[i]  = regime_pred;
    }

    return rates_total;
}
//+------------------------------------------------------------------+
