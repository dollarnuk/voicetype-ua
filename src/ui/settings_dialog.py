"""Settings dialog - Modern sidebar-based configuration UI."""

from typing import Optional

from loguru import logger

try:
    from PyQt6.QtWidgets import (
        QDialog,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QFormLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QCheckBox,
        QPushButton,
        QKeySequenceEdit,
        QSizePolicy,
        QSpacerItem,
        QStackedWidget,
        QFrame,
    )
    from PyQt6.QtCore import pyqtSignal, Qt
    from PyQt6.QtGui import QKeySequence

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from .styles import MAIN_STYLE


class SettingsDialog(QDialog):
    """Settings dialog with modern sidebar navigation.

    Provides UI for editing all application settings. Values are read from
    and written to ConfigManager using dot-notation keys.

    Signals:
        settings_changed: Emitted after settings are saved successfully.
    """

    settings_changed = pyqtSignal()

    def __init__(
        self,
        config_manager,
        audio_capture=None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the settings dialog.

        Args:
            config_manager: ConfigManager instance with get/set/save methods.
            audio_capture: Optional AudioCapture instance for device enumeration.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self._config = config_manager
        self._audio_capture = audio_capture

        self.setWindowTitle("WispanTalk - Налаштування")
        self.setFixedSize(650, 500)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        
        # Apply global style
        self.setStyleSheet(MAIN_STYLE)

        self._sidebar_buttons = []
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the full dialog layout with sidebar and stack."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        sidebar_frame = QFrame()
        sidebar_frame.setObjectName("sidebarFrame")
        sidebar_frame.setFixedWidth(180)
        sidebar_frame.setStyleSheet("background-color: #1E1E1E; border-right: 1px solid #334155;")
        
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(5)

        app_title = QLabel("WispanTalk")
        app_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #6366F1; margin-bottom: 20px; padding-left: 5px;")
        sidebar_layout.addWidget(app_title)

        nav_label = QLabel("Навігація")
        nav_label.setObjectName("sidebarLabel")
        sidebar_layout.addWidget(nav_label)

        self._create_sidebar_item("Загальне", 0, sidebar_layout)
        self._create_sidebar_item("Аудіо", 1, sidebar_layout)
        self._create_sidebar_item("Транскрипція", 2, sidebar_layout)
        self._create_sidebar_item("Гарячі клавіші", 3, sidebar_layout)
        self._create_sidebar_item("Інтерфейс", 4, sidebar_layout)

        sidebar_layout.addStretch()
        
        main_layout.addWidget(sidebar_frame)

        # --- Content Area ---
        content_container = QVBoxLayout()
        content_container.setContentsMargins(0, 0, 0, 0)
        content_container.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.setObjectName("contentArea")
        
        self._stack.addWidget(self._build_general_tab())
        self._stack.addWidget(self._build_audio_tab())
        self._stack.addWidget(self._build_transcription_tab())
        self._stack.addWidget(self._build_hotkeys_tab())
        self._stack.addWidget(self._build_interface_tab())
        
        content_container.addWidget(self._stack)

        # --- Bottom Buttons ---
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background-color: #121212; border-top: 1px solid #334155; padding: 10px;")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(20, 10, 20, 10)
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        self._btn_cancel = QPushButton("Скасувати")
        self._btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self._btn_cancel)

        self._btn_save = QPushButton("Зберегти зміни")
        self._btn_save.setObjectName("primaryBtn")
        self._btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self._btn_save)

        content_container.addWidget(btn_frame)
        main_layout.addLayout(content_container)

    def _create_sidebar_item(self, text: str, index: int, layout: QVBoxLayout):
        btn = QPushButton(text)
        btn.setObjectName("sidebarBtn")
        btn.setCheckable(True)
        btn.clicked.connect(lambda: self._on_sidebar_click(index))
        layout.addWidget(btn)
        self._sidebar_buttons.append(btn)
        
        if index == 0:
            btn.setChecked(True)
            btn.setProperty("active", "true")

    def _on_sidebar_click(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._sidebar_buttons):
            is_active = (i == index)
            btn.setChecked(is_active)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _create_section_header(self, title: str, subtitle: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("headerTitle")
        s = QLabel(subtitle)
        s.setObjectName("headerSub")
        layout.addWidget(t)
        layout.addWidget(s)
        return layout

    # --- Tab 1: General ---------------------------------------------------

    def _build_general_tab(self) -> QWidget:
        page = QWidget()
        main_v = QVBoxLayout(page)
        main_v.setContentsMargins(30, 30, 30, 30)
        
        main_v.addLayout(self._create_section_header("Загальні налаштування", "Керування базовою поведінкою WispanTalk"))

        form = QFormLayout()
        form.setSpacing(15)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._combo_language = QComboBox()
        self._combo_language.addItem("Українська", "uk")
        self._combo_language.addItem("English", "en")
        self._combo_language.addItem("Автоматично", "auto")
        form.addRow("Основна мова:", self._combo_language)

        self._combo_input_mode = QComboBox()
        self._combo_input_mode.addItem("Push-to-Talk (утримання)", "ptt")
        self._combo_input_mode.addItem("Toggle (перемикання)", "toggle")
        form.addRow("Режим активування:", self._combo_input_mode)

        self._chk_start_minimized = QCheckBox("Запускати згорнутим у трей")
        form.addRow("", self._chk_start_minimized)

        self._chk_start_with_windows = QCheckBox("Запускати разом з Windows")
        form.addRow("", self._chk_start_with_windows)

        main_v.addLayout(form)
        main_v.addStretch()
        return page

    # --- Tab 2: Audio ------------------------------------------------------

    def _build_audio_tab(self) -> QWidget:
        page = QWidget()
        main_v = QVBoxLayout(page)
        main_v.setContentsMargins(30, 30, 30, 30)

        main_v.addLayout(self._create_section_header("Налаштування аудіо", "Вибір мікрофона та параметрів запису"))

        form = QFormLayout()
        form.setSpacing(15)

        self._combo_device = QComboBox()
        self._combo_device.addItem("Системний за замовчуванням", None)
        if self._audio_capture is not None:
            try:
                devices = self._audio_capture.get_input_devices()
                for dev in devices:
                    self._combo_device.addItem(dev.get("name", "Unknown"), dev.get("index"))
            except Exception as e:
                logger.error(f"Device enumeration failed: {e}")
        form.addRow("Мікрофон:", self._combo_device)

        self._combo_sample_rate = QComboBox()
        for rate in (16000, 44100, 48000):
            self._combo_sample_rate.addItem(f"{rate/1000} kHz", rate)
        form.addRow("Частота дискретизації:", self._combo_sample_rate)

        main_v.addLayout(form)
        main_v.addStretch()
        return page

    # --- Tab 3: Transcription ----------------------------------------------

    def _build_transcription_tab(self) -> QWidget:
        page = QWidget()
        main_v = QVBoxLayout(page)
        main_v.setContentsMargins(30, 30, 30, 30)

        main_v.addLayout(self._create_section_header("Deepgram API", "Налаштування хмарного розпізнавання"))

        form = QFormLayout()
        form.setSpacing(15)

        self._edit_dg_key = QLineEdit()
        self._edit_dg_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._edit_dg_key.setPlaceholderText("Вставте ключ...")
        form.addRow("API Key:", self._edit_dg_key)

        self._chk_streaming = QCheckBox("Стрімінг у реальному часі")
        form.addRow("", self._chk_streaming)

        main_v.addLayout(form)
        
        info = QLabel("Deepgram забезпечує найвищу точність та швидкість.\nРеєстрація доступна на deepgram.com")
        info.setStyleSheet("color: #94A3B8; font-size: 9pt; margin-top: 20px;")
        main_v.addWidget(info)
        
        main_v.addStretch()
        return page

    # --- Tab 4: Hotkeys ----------------------------------------------------

    def _build_hotkeys_tab(self) -> QWidget:
        page = QWidget()
        main_v = QVBoxLayout(page)
        main_v.setContentsMargins(30, 30, 30, 30)

        main_v.addLayout(self._create_section_header("Гарячі клавіші", "Налаштування швидкого доступу"))

        form = QFormLayout()
        form.setSpacing(15)

        self._key_ptt = QKeySequenceEdit()
        form.addRow("Push-to-talk:", self._key_ptt)

        self._key_toggle_lang = QKeySequenceEdit()
        form.addRow("Зміна мови (UA/EN):", self._key_toggle_lang)

        main_v.addLayout(form)
        main_v.addStretch()
        return page

    # --- Tab 5: Interface --------------------------------------------------

    def _build_interface_tab(self) -> QWidget:
        page = QWidget()
        main_v = QVBoxLayout(page)
        main_v.setContentsMargins(30, 30, 30, 30)

        main_v.addLayout(self._create_section_header("Інтерфейс", "Візуальні ефекти та сповіщення"))

        form = QFormLayout()
        form.setSpacing(15)

        self._combo_theme = QComboBox()
        self._combo_theme.addItem("Modern Dark (Indigo)", "dark")
        self._combo_theme.addItem("Light (Classic)", "light")
        form.addRow("Тема програми:", self._combo_theme)

        self._chk_overlay = QCheckBox("Показувати оверлей під час запису")
        form.addRow("", self._chk_overlay)

        self._chk_sounds = QCheckBox("Відтворювати звукові сигнали")
        form.addRow("", self._chk_sounds)

        main_v.addLayout(form)
        main_v.addStretch()
        return page

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load_values(self) -> None:
        """Populate all widgets from ConfigManager."""
        cfg = self._config

        # General
        self._set_combo_by_data(self._combo_language, cfg.get("general.language", "uk"))
        self._set_combo_by_data(self._combo_input_mode, cfg.get("hotkeys.input_mode", "ptt"))
        self._chk_start_minimized.setChecked(cfg.get("general.start_minimized", True))
        self._chk_start_with_windows.setChecked(cfg.get("general.start_with_windows", False))

        # Audio
        self._set_combo_by_data(self._combo_device, cfg.get("audio.device_index", None))
        self._set_combo_by_data(self._combo_sample_rate, cfg.get("audio.sample_rate", 16000))

        # Transcription
        self._edit_dg_key.setText(cfg.get("transcription.deepgram_api_key", ""))
        self._chk_streaming.setChecked(cfg.get("transcription.streaming", True))

        # Hotkeys
        ptt_seq = cfg.get("hotkeys.push_to_talk", "ctrl+space")
        lang_seq = cfg.get("hotkeys.toggle_language", "ctrl+shift+l")
        self._key_ptt.setKeySequence(QKeySequence.fromString(ptt_seq))
        self._key_toggle_lang.setKeySequence(QKeySequence.fromString(lang_seq))

        # Interface
        self._set_combo_by_data(self._combo_theme, cfg.get("ui.theme", "dark"))
        self._chk_overlay.setChecked(cfg.get("ui.show_status_indicator", True))
        self._chk_sounds.setChecked(cfg.get("ui.play_sounds", True))

    def _on_save(self) -> None:
        """Write current widget values to ConfigManager and close."""
        cfg = self._config

        # General
        cfg.set("general.language", self._combo_language.currentData())
        cfg.set("hotkeys.input_mode", self._combo_input_mode.currentData())
        cfg.set("general.start_minimized", self._chk_start_minimized.isChecked())
        cfg.set("general.start_with_windows", self._chk_start_with_windows.isChecked())

        # Audio
        cfg.set("audio.device_index", self._combo_device.currentData())
        cfg.set("audio.sample_rate", self._combo_sample_rate.currentData())

        # Transcription
        cfg.set("transcription.deepgram_api_key", self._edit_dg_key.text())
        cfg.set("transcription.streaming", self._chk_streaming.isChecked())

        # Hotkeys
        cfg.set("hotkeys.push_to_talk", self._key_ptt.keySequence().toString().lower())
        cfg.set("hotkeys.toggle_language", self._key_toggle_lang.keySequence().toString().lower())

        # Interface
        cfg.set("ui.theme", self._combo_theme.currentData())
        cfg.set("ui.show_status_indicator", self._chk_overlay.isChecked())
        cfg.set("ui.play_sounds", self._chk_sounds.isChecked())

        cfg.save()
        logger.info("Settings saved")

        self.settings_changed.emit()
        self.accept()

    @staticmethod
    def _set_combo_by_data(combo: "QComboBox", value) -> None:
        """Select a QComboBox item whose userData matches *value*."""
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return
