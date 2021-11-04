class NodeCollection(object):
    
    def __init__(self, label, model, number, params):
        self.label = label
        self.number = number

    def __len__(self):
        return self.number

