
#Q1 Write a program to count the number of vowels and consonants present in an input string.
def count_vowels_consonants(text):
    vowels="aeiouAEIOU"
    vowel_count=0
    consonant_count=0
    for char in text:
        if char.isalpha():
            if char in vowels:
                vowel_count +=1
            else:
                consonant_count +=1

    print(f"vowels:{vowel_count}, cononants:{consonant_count}")

user_input=input("Enter the text: ")
count_vowels_consonants(user_input)

#Q2 Write a program that accepts two matrices A and B as input and returns their product AB. Check if A& B are multipliable; if not, return error message.
def multiply_matrices(A,B):
    r1=len(A)
    c1=len(A[0])
    r2=len(B)
    c2=len(B[0])

    if c1!=r2:
        return("The given matrices cannot be multiplied sorryyyyy:(.....bruh ")

    result = [[0 for _ in range(c2)] for _ in range(r1)]
    for i in range(r1):
         for j in range(c2):
             for k in range(c1):
                 result[i][j] += A[i][k]*B[k][j]
    return result
A=[[1,2,3],[2,3,4],[3,4,5]]
B=[[4,5,6],[5,6,7],[6,7,8]]
product=multiply_matrices(A,B)
print("after multiplying:",product)

#Q3 Write a program to find the number of common elements between two lists. The lists conatin integers.
a = [1, 2, 3, 4, 5]
b = [4, 5, 6, 7, 8]

common = list(set(a) & set(b))
print(common)

#Q4 Write a program that accepts a matrix as input and returns its transpose.
def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transpose = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transpose[j][i] = matrix[i][j]
    
    return transpose
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("Original matrix:")
for row in matrix:
    print(row)

transposed_matrix = transpose_matrix(matrix)
print("Transposed matrix:")
for row in transposed_matrix:
    print(row)
    
    

    
