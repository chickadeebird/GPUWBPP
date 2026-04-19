# GPUStacker — GPU-Accelerated Weighted Batch PreProcessing

## 1. General Description

GPUStacker is a GPU-accelerated astrophotography image calibration, registration, and stacking pipeline built for use in PixInsight, although it does also have an experimental standalone GUI. It augments/complements the standard Weighted Batch PreProcessing (WBPP) workflow with a CUDA-optimized pipeline that dramatically reduces processing time while producing comparable or identical output quality. The pipeline is designed for **monochrome camera data** at this time — it processes single-channel FITS and XISF frames acquired through narrowband or broadband filters (L, R, G, B, Ha, OIII, SII, etc.) and produces per-filter stacked master .xisf images ready for color combination in PixInsight. At the present time is runs on Windows and is optimized for NVIDIA CUDA-based GPUs.

The system consists of two components: **AstroPipeline**, a standalone executable that performs the actual image processing using NVIDIA GPU acceleration via CUDA, and **GPUStacker.js**, a PixInsight script that provides a graphical front-end for configuring and launching AstroPipeline from within PixInsight. The executable handles the entire workflow from raw calibration frames to final stacked output — master bias/dark/flat generation, per-frame calibration, star-based image registration with homography fitting, local normalization, optional satellite trail removal, PSF-based frame weighting, sigma-clipped mean stacking, and a final auto-crop.

Timing tests below were done using a typical imaging session with an ASI6200 camera (9576 × 6388 pixels) produces the following dataset (an alternative dataset using images from an STF8300 were also compared):

| Frame Type | Count |
|---|---|
| Bias | 128 |
| Dark | 40 |
| Flat (per filter: L, R, G, B) | 30 each (120 total) |
| Light R, G, B | 14 each |
| Light L | 38 |

Processing this dataset through the complete pipeline (master generation, calibration, registration, local normalization, and stacking for all four filters) yields the following timings on a system with an NVIDIA RTX-class GPU (GEFORCE 1660 6GB):

| Method | Total Processing Time |
|---|-----------------------|
| PixInsight native WBPP (CPU) | **1 hour 58 minutes (118 minutes)** |
| GPUStacker / AstroPipeline | **14 minutes**        |

GPUStacker achieves this speedup by performing nearly every computationally intensive operation on the GPU: background estimation, star detection, centroid computation, image warping (Lanczos-3 interpolation with homography projection), local normalization statistics, and the final sigma-clipped mean stack are all implemented as custom CUDA kernels or optimized CuPy operations.

The pipeline also includes an optional **satellite trail removal** stage that detects and masks satellite and aircraft trails prior to final stacking. This feature is particularly valuable for imaging sessions with fewer subexposures, where a single bright satellite trail may not be adequately rejected by standard sigma-clipping, or for datasets where suboptimal rejection conditions (low frame count, high noise, variable seeing) make statistical outlier rejection unreliable. The trail detection uses an algorithm to accurately identify linear features while preserving baseline image structure.

---

## 2. Installation

### PixInsight Script Installation

To automatically install and receive script updates in PixInsight, add the following URL to **Resources > Updates > Manage Repositories**:

```
https://raw.githubusercontent.com/chickadeebird/GPUStacker/main/
```

After this has been added to the repositories, select **Resources > Updates > Check for Updates**. The new GPUStacker script should appear under **Scripts > ChickadeeScripts**.

### AstroPipeline Executable

The GPUStacker script calls an external executable (**AstroPipeline.exe** on Windows) that performs the actual GPU-accelerated image processing. The executable can be downloaded from the following link:

**Windows (NVIDIA GPU required for speed optimizations):**

```
https://drive.google.com/file/d/13WsyvtQUB95TwFeFFRZhxwdENl_HUIY9/view?usp=sharing
```

An executable for macOS and Linux versions is not available at this time.

**System requirements:**
- NVIDIA GPU with CUDA Compute Capability 6.0 or later (GTX 1000 series and newer)
- NVIDIA GPU driver version 525.0 or later
- CUDA Toolkit 12.x installed on the system (for runtime header access)
- Windows 10/11 (64-bit)

**Setup:**

1. Download the zip file and extract its contents to a folder on your computer (e.g., `C:\AstroPipeline\`).
2. Launch PixInsight and open the GPUStacker script from **Scripts > ChickadeeScripts > GPUStacker**.
3. Click the **wrench icon** at the bottom left of the GPUStacker dialog box.
4. Navigate to the folder where the AstroPipeline executable has been placed and select it. This enables the GPUStacker script to find the executable in the correct location.
5. The executable location is saved and persists across PixInsight sessions.

---

## 3. Detailed Description

### Master Frame Stacking

The pipeline can build master calibration frames (bias, dark, flat) using a GPU-accelerated stacking engine. Individual calibration frames are loaded from the specified directories, optionally filtered by FITS header keywords (IMAGETYP, FILTER), and stacked using a per-pixel winsorized sigma-clipping rejection algorithm. The winsorized clipping iteratively replaces outlier pixel values with the clipped boundary value before recomputing the mean, which is more robust than simple sigma-clipping for small sample sizes (typical of some calibration frame counts). The stacking kernel processes frames in row-slabs that fit within the GPU's VRAM budget, automatically tiling large images (such as the 9576 × 6388 ASI6200 sensor) across multiple passes if the available VRAM is limited. Built master frames are cached to disk so that subsequent pipeline runs can skip the stacking step entirely (controlled by the "Use masters if available" option).

### Flat Frame Flux Equalization

When the "Equalize flat fluxes" option is enabled, each individual flat frame is scaled to a common median value before the master flat is stacked. This corrects for brightness variations between individual flat exposures that arise from changing ambient light conditions during twilight flat acquisition or from slight shutter speed variations in panel-illuminated flats. Without equalization, the master flat can exhibit low-frequency brightness gradients that propagate into the calibrated light frames as systematic background errors. The equalization computes each frame's median on a strided subsample (for speed), then scales the frame by the ratio of the global median to the per-frame median. The operation is performed in-place on the GPU to minimize memory overhead.

### Image Calibration

Each light frame is calibrated by subtracting the master bias, subtracting the exposure-time-scaled master dark, and dividing by the normalized master flat. The exposure time for each light frame and the master dark are read from the FITS header (`EXPTIME` keyword). The dark-scaling ratio is computed per-frame as `light_exposure / dark_exposure`, which handles mixed-exposure datasets. The master flat is normalized to its median value before division so that the calibrated output preserves the original photometric scale of the light frames. The flat division includes a floor clamp to prevent division by near-zero values in vignetted corners. All calibration arithmetic is performed in float32 on the GPU, with the calibrated frame written back to CPU memory via direct DMA transfer into a pinned host buffer.

### Image Registration

Star-based image registration aligns all light frames to a common reference frame using a multi-stage mini pipeline:

1. **Star detection**: Background estimation via block-mean subtraction, noise estimation via MAD on a subsampled residual, peak finding via local-maximum filtering above a sigma threshold, and sub-pixel centroid refinement via a flux-weighted 11×11 CUDA kernel.

2. **Star matching**: Triangle asterism features (scale-invariant side ratios) are computed for the brightest stars in each frame and matched against the reference frame's features using a KD-tree nearest-neighbor search. A GPU-accelerated brute-force tiled CUDA kernel provides the fastest matching path when CuPy is available, which is compiled in the AstroPipeline.exe executable. A translation-voting pre-filter and spatial nearest-neighbor fallback ensure robust matching even for cross-filter images with different star brightness distributions.

3. **Transform estimation**: A 4-point homography (8-DOF perspective model) is fitted via RANSAC with normalized DLT, adaptive iteration count, and iterative inlier refinement. The homography correctly handles field rotation, differential flexure, and slight optical distortion. Transform validation checks reject degenerate solutions (reflections, extreme scale, high condition number, excessive perspective coefficients).

4. **Image warping**: A custom CUDA kernel performs Lanczos-3 (6×6 tap) interpolation with the fitted inverse homography, computing source coordinates in double precision to avoid sub-pixel positioning errors at large image dimensions. The kernel processes all output pixels in parallel and reuses a single GPU output buffer across frames.

### Local Normalization

Local normalization equalizes the per-pixel background level and noise scale across all frames within each filter, compensating for frame-to-frame variations in sky brightness, transparency, and gradient structure. For each frame, the local mean and local standard deviation are computed over a 257×257 pixel (radius=128) sliding window using separable box filtering via cumulative sums — an O(H×W) algorithm regardless of window size.

A critical refinement is the **masked statistics** approach: bright pixels (above `median + 2σ`, estimated via MAD) are completely excluded from the local statistics computation rather than replaced with a constant value. This prevents satellite trails and bright stars from contaminating the local mean and standard deviation in their neighborhood. Three box-filter passes (masked image sum, mask count, masked image-squared sum) yield the masked local mean and variance. The normalization formula `out = (frame − local_mean) × (ref_std / frame_std) + ref_mean` maps each frame's local distribution to match the reference frame, using an additive-only fallback in regions where either standard deviation falls below a noise floor.

The pipeline automatically selects the **cleanest reference frame** by counting bright outlier pixels across all frames and choosing the one with the fewest. This prevents a frame containing a prominent satellite trail from being used as the normalization target, which would otherwise imprint a faint trail-shaped artifact across every normalized output.

### Satellite Trail Removal

The optional satellite trail removal stage detects and masks satellite and aircraft trails in the preconditioned residual images prior to final stacking. Although the logging suggests that Hough transforms were used for autodetection of lines, this approach was abandoned for a novel heuristic implementation.

### PSF-Based Frame Weighting

When the "PSF Signal Weight" option is enabled, each frame receives a quality weight proportional to its signal-to-noise ratio, star sharpness, and star roundness. The weight is computed as `SNR² × FWHM_score / eccentricity`, where SNR is estimated from the frame's median and MAD, FWHM_score penalizes frames with poor seeing, and eccentricity penalizes frames with elongated stars (from tracking errors or wind shake). Higher-quality frames contribute more to the final stack, improving the effective resolution and SNR of the output.

### Final Stacking

The final per-filter stack combines all calibrated, registered, locally normalized frames using a weighted mean with winsorized sigma-clipping rejection. The rejection iteratively identifies and clips pixels whose values deviate by more than κ standard deviations from the local mean (default κ = 3.0 for both low and high bounds). Satellite trail masks, when available, are passed as pre-rejection masks that unconditionally exclude trail pixels from the combination regardless of their statistical properties. The stacking kernel processes the image in row-slabs sized to fit within the GPU's VRAM budget, supporting arbitrarily large images on GPUs with limited memory.

---

## 4. GPUStacker PixInsight Script

The GPUStacker.js script provides a graphical interface within PixInsight for configuring and launching the AstroPipeline executable. The dialog is organized into three sections:

### Frame Directories

Five directory selectors allow the user to specify the locations of bias, dark, flat, light, and output (save) frames. Each selector includes a text field and a browse button. The save directory is optional — if left empty, output files are saved to a `masters` subdirectory within the light frame directory. All directory paths are persisted across PixInsight sessions via a configuration file. The default execution options auto-detect for the existence of an NVIDIA GPU with CUDA installed, and auto-optimizes to use ~92% of available free VRAM.

### Options

- **Master Frames**: "Use masters if available" (default) skips master generation when cached masters exist; "Recalculate all masters" always rebuilds from raw calibration frames.
- **Stacking Weight**: PSF Signal Weight (default) or None.
- **Equalize flat fluxes**: Scales individual flat frames to a common median before master flat generation.
- **Autocrop stacked images**: Crops the final stack to the rectangular intersection covered by all registered frames.
- **Save registered frames**: Saves each individually registered frame (prior to local normalization) to disk for inspection.

### Satellite Trail Removal

- **Enable satellite trail removal**: Activates the trail detection and masking stage before final stacking.
- **Save intermediate files**: Saves the preconditioned, median-subtracted, and mask images for inspection and tuning.
- **Line width (px)**: Width of the mask drawn over detected trails (default 7 pixels).
- **Min line size (% diag)**: Minimum trail length as a percentage of the image diagonal.
- **Range threshold mult**: Detection threshold expressed as a multiple of the per-frame residual median. Default is 3.0. Lower numbers include fainter lines but may also erroneously include other image structures.

### Running the Pipeline

Clicking the red "Create masters, ..." button writes a platform-appropriate launcher script (batch file on Windows, shell script on macOS/Linux) and executes AstroPipeline in a separate console window. The PixInsight interface remains responsive while the pipeline runs. Progress and timing information are displayed in the console window.

---

## 5. Future Directions

### Platform Support

The AstroPipeline executable is currently available for **Windows** only, as the build and execution environment uses NVIDIA CUDA on Windows. Future releases may include:

- **Linux**: The pipeline's Python codebase and CUDA kernels are platform-independent. A Linux build would require packaging with PyInstaller on a Linux system with the CUDA toolkit installed. The GPUStacker.js script already includes Linux-compatible launcher script generation.
- **macOS**: Apple Silicon Macs do not support NVIDIA CUDA, so a macOS port would require either a Metal Compute or OpenCL backend (replacing CuPy's CUDA kernels) or a CPU-only mode. The pipeline's CPU fallback paths are functional but significantly slower. macOS with external NVIDIA eGPUs (rare) could theoretically use the existing CUDA code.

### Color / RGB Image Support

The current pipeline processes **monochrome (single-channel) frames** only. Each filter (L, R, G, B, narrowband) is calibrated, registered, and stacked independently, producing single-channel master images that the user combines into a color image using PixInsight's standard tools (ChannelCombination, LRGBCombination, PixelMath).

A future release may implement native **three-plane (RGB) stacking** for one-shot color cameras (OSC/DSLR), which would:

- Debayer raw Bayer-pattern frames using GPU-accelerated interpolation
- Register and stack all three color channels simultaneously
- Apply per-channel calibration with color-aware flat correction
- Produce a ready-to-process RGB color image directly from the pipeline

This would extend GPUStacker's applicability to the large community of astrophotographers using one-shot color cameras.

### Additional Planned Improvements

- **Drizzle integration**: Sub-pixel resampling during stacking to recover resolution beyond the native pixel scale, particularly beneficial for undersampled optical systems.
- **Comet/asteroid stacking mode**: Registration on a moving target rather than fixed stars, enabling comet and asteroid imaging without trailing.
- **Mosaic support**: Multi-panel registration and stacking for wide-field mosaic imaging projects.
