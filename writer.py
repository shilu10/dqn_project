class Writer:
    def __init__(self, fname): 
        self.fname = fname 

    def write_to_file(self, content): 
        with open(self.fname, "wb") as file: 
            file.write(content)

    def read_file(self, fname):
        with open(fname, "rb") as file: 
            return file.read()
            
             
