# Third-party notices

- **MediaPipe** — Copyright The MediaPipe Authors. Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
  `js/handTracker.js` contains JavaScript ports, modified for hand landmarks, of
  `mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc`, `procrustes_solver.cc` and the
  one-euro landmark smoothing in `mediapipe/calculators/util/landmarks_smoothing_calculator_utils.cc`.
  The MediaPipe Tasks Vision library itself is loaded from a CDN.
- **OneEuroFilter** (`js/vendor/OneEuroFilter.js`) — Copyright 2019 Inria, Géry Casiez. BSD 3-Clause; license text is in the file header.
- **svd-js** (`js/vendor/svd.js`) — Copyright 2017 Danilo Salvati. MIT; license text is in the file header.
- **Three.js** r170 — Copyright 2010-2024 Three.js Authors. MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE). Loaded from a CDN.
- **thin-plate-spline** (`js/vendor/thin-plate-spline.js`) — Copyright 2026 pravoobi. MIT; license text is in the file header.
- Native app (`ios/WoundAR/HandMath.swift`) contains Swift ports of the OneEuroFilter (BSD-3-Clause, Inria / Géry Casiez) and of the thin-plate-spline package's linear solver (MIT, pravoobi) listed above. Its skin tracker implements the published pyramidal Lucas–Kanade algorithm (J.-Y. Bouguet, "Pyramidal Implementation of the Affine Lucas Kanade Feature Tracker", Intel, 2000) from the paper; no third-party code. The camera background in `ios/WoundAR/Renderer.swift` / `Shaders.metal` follows Apple's ARKit Metal app template (Xcode "Augmented Reality App", Metal content: Y/CbCr textures via CVMetalTextureCache and its YCbCr → RGB matrix), rewritten. The deformable skin mesh follows the published method of Pilet, Lepetit & Fua (IJCV 2008) and the shading-ratio idea of Bradley, Roth & Bose ("Augmented reality on cloth with realistic illumination", Machine Vision and Applications, 2009); own code.
