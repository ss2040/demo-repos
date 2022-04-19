# serialize.py
import yaml

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f'<Person: {self.name} - {self.age}>'

    def __repr__(self):
        return str(self)

person = Person('Dhruv', 24)
with open('person.yml', 'w') as output_file:
    yaml.dump(person, output_file)
