class Subject:
    '''Represents a subject in a research study.

    Attributes:
        sid(int): a unique identifier for the sibject
        name(str): the subject's name
        measurements(dict of str:float): represents the subjects timestamped
            measurements throughout the day
        
        num_subjects(int): class-level attribute
            total number of subjects in the study
    '''

    num_subjects = 0 # class-level attribute
    # meaning there is only one num_subjects variable
    # shared across all subject objects

    # __init__() special method
    # for initializing an object (like a construtor)
    def __init__(self,name, measurements=None):
        # self if like this
        # self refers to the "current" or "invoking" object
        self.sid = Subject.num_subjects
        Subject.num_subjects += 1
        self.name = name
        if measurements is None:
            measurements = {}
        self.measurements = measurements
    
    # __str__() special method
    # for converting an object to a string for printing
    def __str__(self):
        return "SID: " + str(self.sid) + " Name: " + self.name  + " Measurements: " + str(self.measurements)

    # non-special methods
    def record_measurement(self,timestamp, value):
        # Should do error checking but for now...
        self.measurements[timestamp] = value


sub1 = Subject("Bob",{1:2,3:4})
print(sub1)
print(sub1.num_subjects)
print(sub1.name)
print(sub1.measurements)