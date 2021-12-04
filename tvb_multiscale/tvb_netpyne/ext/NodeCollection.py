class NodeCollection(object):
    
    def __init__(self, brain_region, pop_label, size):
        self.brain_region = brain_region
        self.pop_label = pop_label
        self.size = size
        
        self.label = brain_region + '.' + pop_label

    def __len__(self):
        return self.size

