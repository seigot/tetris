class ArtManager:
    def __init__(self):
        self.art_elements = []

    def add_art_element(self, art_element):
        self.art_elements.append(art_element)

    def get_art_elements(self):
        return self.art_elements
