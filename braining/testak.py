# import re
#
#
# def extract_response_from_template(template):
#     # Define a regular expression to capture the desired response
#     pattern = re.compile(r'### Response:\s*([^#]+?)\s*### End:', re.DOTALL)
#
#     # Find the first match in the template
#     match = re.search(pattern, template)
#
#     # Extract the match if found
#     if match:
#         extracted_response = match.group(1).strip()
#         return extracted_response
#     else:
#         return None
#
#
# def extract_response_from_output(generated_output):
#     # Define a regular expression to extract the desired response
#     pattern = re.compile(r'### Response: (.*?) ### End Response')
#
#     # Find all matches in the generated output
#     matches = re.findall(pattern, generated_output)
#
#     # Extract the first match (assuming there is only one response in your case)
#     if matches:
#         extracted_response = matches[0]
#         print("Extracted Response:", extracted_response)
#         return extracted_response
#     else:
#         print("No match found.")
#         return None
#
#
# if __name__ == '__main__':
#     #     generated_output = """
#     #     ### ### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.
#     # ### Input Question:
#     #  Three doors are present. Every door has a way to exit. The first door has lava and fire inside. Lions behind the second door haven't eaten in five years. There is a 1,000-foot drop into an alligator inside the third door. Which door will you probably get through?
#     # ### ChoiceList:
#     #  ["The third door, the alligator can be tierd and won't eat people.", "The second door with the lions. Since they haven't eaten in 5 years they will have died.", 'The first door, the first door is destroyed and you can see the fire in it.', 'None of above.']
#     # ### Response:
#     #  The second door with the lions. Since they haven't eaten in 5 years they will have died.  ### End ### Response:
#     #  The second door with the lions. Since they haven't eaten in 5 years they will have died.  ### End ### Response:
#     #   ### End ### Response:
#     #   ### End ### Response:
#     #   ### End ### Response:
#     #   ### End ### Response:
#     #   ### End ### Response"""
#     generated_output2 = """
#     ### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.
# ### Input Question:
#  Three doors are present. Every door has a way to exit. The first door has lava and fire inside. Lions behind the second door haven't eaten in five years. There is a 1,000-foot drop into an alligator inside the third door. Which door will you probably get through?
# ### ChoiceList:
#  ["The third door, the alligator can be tierd and won't eat people.", "The second door with the lions. Since they haven't eaten in 5 years they will have died.", 'The first door, the first door is destroyed and you can see the fire in it.', 'None of above.']
# ### Response:
#  The second door with the lions. Since they haven't eaten in 5 years they will have died.  ### End ### Response:
#  The second door with the lions. Since they haven't eaten in 5 years they will have died.  ### End ### Response:
#   ### End ### Response:
#   ### End ### Response:
#   ### End ### Response:
#   ### End ### Response:
#   ### End ### Response"""
#     # # Use regular expression to extract the response
#     # response_match = re.search(r'### Response:(.*?)### End ###', generated_output, re.DOTALL)
#     #
#     # # Check if there is a match and extract the response
#     # if response_match:
#     #     extracted_response = response_match.group(1).strip()
#     #     print(extracted_response)
#     # else:
#     #     print("No response found.")
#
#     # Use regular expression to extract the response
#     response_matches = re.findall(r'### Response:(.*?)### End ###', generated_output2, re.DOTALL)
#
#     # Remove duplicates and unnecessary characters
#     unique_responses = set(response.strip() for response in response_matches)
#
#     # Print the unique responses
#     for extracted_response in unique_responses:
#         print("extracted_response", extracted_response)
#
#     # Example usage
#     template = """
#      ### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.Avoid generating duplicate tokens and aim for clarity in your answer.
#     ### Input Question:
#      A man moved into a new house and decided he was tired after the journey and turned off all the lights and went to bed. The next morning he killed 365 people, he does not sleep walk he slept like a baby. How could this be?
#     ### ChoiceList:
#      ['The man could be a calendar maker. He accidentally excluding February 29th on a leap year\'s calendar before sleep. This "killed" the day for 365 people.', 'The man is a world-renowned botanist who has just moved into a new house with a rare greenhouse containing 365 unique and endangered plant species.', 'He lived in a light house.', 'None of above.']
#     ### Response:   ### End:
#      He lived in a light house.  ### Time Taken:
#      1 second  ### Response:
#      He lived in a light house.  ### End:
#      None of above.  ### Time Taken:
#      None of above.  ### Response:
#      He lived in a light house.  ### End:
#      None of above.  ### Response:
#      He lived in a light house.  ### End:
#     """
#
#     extracted_response = extract_response_from_template(template)
#     print("Extracted response:", extracted_response)
import re

def extract_response_from_template(template):
    # Define a regular expression to capture the desired response
    pattern = r"### End:(.*?)### Time Taken:"
    matches = re.search(pattern, template, re.DOTALL)
    print(matches)
    if matches:
        response = matches.group(1).strip()
        return(response)
    else:
        return("No match found.")

template = """
 ### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.Avoid generating duplicate tokens and aim for clarity in your answer.
### Input Question:  
 A man moved into a new house and decided he was tired after the journey and turned off all the lights and went to bed. The next morning he killed 365 people, he does not sleep walk he slept like a baby. How could this be?
### ChoiceList:  
 ['The man could be a calendar maker. He accidentally excluding February 29th on a leap year\'s calendar before sleep. This "killed" the day for 365 people.', 'The man is a world-renowned botanist who has just moved into a new house with a rare greenhouse containing 365 unique and endangered plant species.', 'He lived in a light house.', 'None of above.']
### Response:   ### End:  
 He lived in a light house.  ### Time Taken:  
 1 second  ### Response:  
 He lived in a light house.  ### End:  

"""

extracted_response = extract_response_from_template(template)
print("Extracted response:", extracted_response)