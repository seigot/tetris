class UIFeedback:
    def __init__(self):
        pass

    def display_ren_and_bonus(self, ren_counter, firepower_bonus):
        # Display the REN counter and firepower bonus on the UI
        print(f"Current REN: {ren_counter}")
        print(f"Firepower Bonus: {firepower_bonus}")

    def update_ui(self, game_manager):
        ren_counter = game_manager.scoring_system.ren_counter
        firepower_bonus = game_manager.scoring_system.get_firepower_bonus()
        self.display_ren_and_bonus(ren_counter, firepower_bonus)

# Example usage
# ui_feedback = UIFeedback()
# ui_feedback.update_ui(game_manager)
