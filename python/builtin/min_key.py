# min(iterable, key=function) returns the item with the minimum value after applying the key function

# Example 1: Finding the shortest string
words = ["apple", "hi", "banana", "ok"]
shortest = min(words, key=len)
print(shortest)  # Output: "hi" (length 2)

# Example 2: Finding the person with minimum age
people = {"Alice": 30, "Bob": 25, "Charlie": 35}
youngest = min(people, key=people.get)
print(youngest)  # Output: "Bob" (age 25)