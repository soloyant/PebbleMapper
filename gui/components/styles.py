"""Shared dark-panel colour vocabulary.

The palette of the embedded log/prompt consoles and the dark inspector/records
panels in the Zonal and Digitize tabs. The app's default NiceGUI theme is
light; these are intentional "terminal / inspector" surfaces, not a global
dark mode. Pure constants; must not import ``gui.app``.
"""

PANEL_BG = "#1e2530"        # dark surface background
PANEL_BORDER = "#3a4555"    # dark surface border
PANEL_TEXT = "#e6e6e6"      # primary text on a dark surface
PANEL_MUTED = "#6b7a8f"     # secondary / placeholder text on dark
PANEL_MUTED_2 = "#8fa0b8"   # tertiary text on dark

__all__ = ["PANEL_BG", "PANEL_BORDER", "PANEL_TEXT", "PANEL_MUTED", "PANEL_MUTED_2"]
