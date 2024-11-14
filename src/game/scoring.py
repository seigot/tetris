class ScoringSystem:
    def __init__(self):
        self.ren_counter = 0
        self.firepower_bonus = 0

    def update_score(self, lines_cleared):
        if lines_cleared > 0:
            self.ren_counter += 1
        else:
            self.ren_counter = 0
        self.calculate_firepower_bonus()

    def calculate_firepower_bonus(self):
        if self.ren_counter <= 1:
            self.firepower_bonus = 0
        elif 2 <= self.ren_counter <= 3:
            self.firepower_bonus = 1
        elif 4 <= self.ren_counter <= 5:
            self.firepower_bonus = 2
        elif 6 <= self.ren_counter <= 7:
            self.firepower_bonus = 3
        elif 8 <= self.ren_counter <= 10:
            self.firepower_bonus = 4
        else:
            self.firepower_bonus = 5

    def get_firepower_bonus(self):
        return self.firepower_bonus

    def reset(self):
        self.ren_counter = 0
        self.firepower_bonus = 0
