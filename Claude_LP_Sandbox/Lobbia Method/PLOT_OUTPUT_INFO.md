# Plot Output Information

## Overview

Both the batch processing script and notebook generate individual plots for each trace file analyzed.

## Output Directory Structure

```
Axial Langmuir/
├── plots/                                    # Created automatically
│   ├── trace_file_1_IV_characteristic.png
│   ├── trace_file_1_Te_semilog_fit.png
│   ├── trace_file_2_IV_characteristic.png
│   ├── trace_file_2_Te_semilog_fit.png
│   └── ...
├── analysis_summary.csv                      # Results table (CSV)
├── trace_file_1.csv                          # Original data
├── trace_file_2.csv
└── ...
```

## Plot Types

### 1. **I-V Characteristic Plot** (`*_IV_characteristic.png`)

Shows the raw probe current vs bias voltage with identified potentials marked.

**Features:**
- **Black line**: Raw measured I-V trace
- **Red dashed line + marker**: Floating potential (Vf)
- **Blue dashed line + marker**: Plasma potential (Vp)
- Voltage range: Full bias voltage sweep
- Current in mA for easy reading

**Use for:** Verifying Vf and Vp identification, visual inspection of data quality, checking for artifacts or noise

### 2. **Semilog Fit Plot** (`*_Te_semilog_fit.png`)

Shows the electron saturation region with the linear fit used to extract electron temperature.

**Features:**
- **Gray dots**: All electron current data (Ie for V ≥ Vf)
- **Blue dots + line**: Fit region (V ∈ [Vf, Vp - 2Te])
- **Red line**: Linear fit on semilog plot (exponential in linear space)
- **Red dashed line**: Vf position
- **Blue dashed line**: Vp position
- Logarithmic y-axis for clarity
- Te value and uncertainty shown in legend

**Use for:**
- Verifying Te extraction quality
- Checking fit region selection
- Confirming exponential behavior of electron saturation
- Assessing data quality in exponential region

## Interpreting the Plots

### Good Quality Data

✓ **I-V curve**
- Smooth transition from ion to electron regions
- No sharp discontinuities or noise spikes
- Clear saturation regions on both ends

✓ **Semilog fit**
- Red fit line overlays blue points well
- Linear relationship on semilog plot
- High R² value (shown in analysis_summary.csv)
- Fit region spans reasonable voltage window

### Potential Issues

✗ **I-V curve**
- Noisy or scattered data
- Multiple oscillations
- Missing or ambiguous saturation regions
- Vf/Vp appear in unexpected locations

✗ **Semilog fit**
- Red line significantly deviates from blue points
- Non-exponential behavior (curved fit region)
- Low R² value
- Fit region too narrow or too broad

## File Naming Convention

Files are named based on the input trace filename:

```
Input:  trace_2024_10A_01.csv
Output: trace_2024_10A_01_IV_characteristic.png
        trace_2024_10A_01_Te_semilog_fit.png
```

The suffix is removed from input filename and replaced with descriptor.

## Quick Inspection Workflow

1. **Run analysis**:
   ```bash
   python process_axial_langmuir.py
   ```

2. **Check CSV results**:
   ```
   Axial Langmuir/analysis_summary.csv
   ```
   - Look for convergence, sheath regime, parameter ranges

3. **Inspect plots for each file**:
   ```
   Axial Langmuir/plots/
   ```
   - Open `*_IV_characteristic.png` to verify Vf/Vp
   - Open `*_Te_semilog_fit.png` to verify Te extraction
   - Look for unexpected patterns or quality issues

4. **Compare across runs**:
   - Save previous `plots/` directory with date prefix if comparing methods
   - Visual comparison of all IV curves gives quick overview of data consistency

## Batch Plotting in Jupyter

If using the notebook, plots are generated after batch analysis:

1. Run **Example 2** cells to process all traces and build results
2. Run **Save Individual Plots** section to generate PNG files
3. Plots appear in `plots/` subdirectory with same naming convention

## Resolution and Format

- **DPI**: 150 (suitable for printing and screen display)
- **Format**: PNG (lossless, good for archival)
- **Size**: ~200 KB per plot (varies with figure complexity)
- **Color**: Full color, suitable for color printing

## Tips

- **Large batches**: With many traces, the plots directory can grow quickly. Consider organizing by date or experimental condition.
- **Archive**: Keep original `plots/` directory if re-running analysis with different parameters for comparison.
- **Sharing**: PNG format is widely supported; plots can be easily included in reports or presentations.
- **Post-processing**: Use standard image tools (ImageMagick, Python PIL, etc.) to batch resize, convert formats, or combine into multi-panel figures.

---

For questions about specific plots or interpretation, refer to the Lobbia & Beal (2017) paper reference in README.md.
