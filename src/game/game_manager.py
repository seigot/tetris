from scoring import ScoringSystem

class GameManager:
    def __init__(self):
        self.scoring_system = ScoringSystem()

    def clear_lines(self, lines_cleared):
        # Existing logic for clearing lines
        # ...
        
        # Update scoring system with the number of lines cleared
        self.scoring_system.update_score(lines_cleared)
        
        # Get the current firepower bonus
        firepower_bonus = self.scoring_system.get_firepower_bonus()
        
        # Apply the firepower bonus to the game state
        # ...

        return firepower_bonus
