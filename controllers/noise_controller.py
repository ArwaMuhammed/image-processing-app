from controllers.main_controller import MainController


class NoiseController:
    def __init__(self, window, image_manager):
        self.ui = window
        self.image_manager = image_manager
        self.noisy_image = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # Noise types
        self.ui.noise_combo_type.addItems(
            ["Gaussian", "Uniform", "Salt & Pepper"]
        )

        # Filters
        self.ui.noise_combo_filter.addItems(
            ["Average (3x3)", "Gaussian (3x3)", "Median (3x3)"]
        )

    def _connect_signals(self):
        # Buttons
        self.ui.noise_btn_apply.clicked.connect(self.apply_noise)
        self.ui.filter_btn_apply.clicked.connect(self.apply_filter)

        # Auto-apply noise when slider changes
        self.ui.noise_slider_amount.valueChanged.connect(self.apply_noise)

    def apply_noise(self):
        image = self.image_manager.original_image
        if image is None:
            return

        # Add noise
        noise_type = self.ui.noise_combo_type.currentText()
        amount = self.ui.noise_slider_amount.value() / 100.0

        from core.noise import add_noise

        self.noisy_image = add_noise(image, noise_type, amount)

        # Display noisy image, expand to fill its group box
        MainController.display_image(self, self.noisy_image, self.ui.noise_noisy_image)

    def apply_filter(self):
        image = self.noisy_image
        if image is None:
            return

        filter_type = self.ui.noise_combo_filter.currentText()

        from core.filters import apply_filter

        filtered_image = apply_filter(image, filter_type)

        # Display filtered image, expand to fill its group box
        MainController.display_image(self, filtered_image, self.ui.noise_filtered_image)
