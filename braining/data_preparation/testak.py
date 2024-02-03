import re
raw_content = """
        "### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. "
        "Please respond after ### Response: "
        "Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. "
        "Pay equal attention to both correct and incorrect options to enhance your selection skills."
        "Avoid generating duplicate tokens and aim for clarity in your answer
"""
star_content = """
### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.Avoid generating duplicate tokens and aim for clarity in your answer.
### Input Question:  
 Take 2 from 5 and leave 4, how is that possible
### ChoiceList:  
 ['Take the 5 and subtract 2 from it. This results in the number 3. Now, remove the vertical top part of the digit "3" and you are left with the digit "4." ', 'FIVE. Remove the 2 letters F and E from five and you have in which is the Roman numeral for four.', 'Imagine you have five fingers on your hand. If you take two fingers (your index and middle fingers) and hold them down, you are left with four fingers remaining upright. So, by taking 2 fingers from the set of 5 fingers, you are left with 4 fingers.', 'None of above.']
### Response:  ### End of question's context. Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.Avoid generating duplicate tokens and aim for clarity in your answer. ### Input Question:  
 Take 2 from 5 and leave 4, how is that possible
 ### ChoiceList:  
 Take the 5 and subtract 2 from it
"""

# Define the regular expression pattern for finding text between markers
pattern = re.compile(r'### Response(.*?)### End', re.DOTALL)

# Find all matches in the raw content
matches = pattern.findall(raw_content)

# Extracted content from each pair of markers in the raw file
extracted_content = [match.strip() for match in matches]

# Replace the corresponding portions in the star content with an empty string
modified_star_content = star_content
for content in extracted_content:
    modified_star_content = modified_star_content.replace(content, '*')
print("\n\n\n\n\n")
print(modified_star_content)