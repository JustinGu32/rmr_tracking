"""Generalist motion-tracking task.

Derived from the popart task but without PopArt critic normalization. Retains
the category-aware multi-clip command (used for adaptive sampling over
categories / clips) and one opt-in jump termination (T3 bad_anchor_pos_z_flight).
"""
