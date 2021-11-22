class NodeCollection(object):
    
    def __init__(self, label, number):
        self.label = label
        self.number = number

    def __len__(self):
        return self.number

