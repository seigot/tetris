class LineClear:
    def __init__(self):
        self.lines_cleared = 0


    def clear_lines(self, lines):
        self.lines_cleared += len(lines)
        if self.is_perfect_clear():
            self.trigger_perfect_clear_effect()

    def is_perfect_clear(self):
        # Logic to determine if the clear is a perfect clear
        pass

    def trigger_perfect_clear_effect(self):
        # Logic to trigger visual effects for perfect clear
        self.effects.play_perfect_clear_effect()
