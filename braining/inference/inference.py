import re

import regex
from sentence_transformers import SentenceTransformer, util


class PostProcess:
    def __init__(self, generated_sample, text_to_ignore, retriever_model_name='paraphrase-MiniLM-L6-v2'):
        self.retriever_model = SentenceTransformer(retriever_model_name)
        self.generated_sample = generated_sample
        self.text_to_ignore = text_to_ignore

    # def retrieve_top_choice(self, choices, top_k):
    #     """
    #     Retrieve the top chunk.
    #     Args:
    #         generated_sample: The query received from the user.
    #         choices: chunk_document function's output.
    #         top_k: The number of top chunks to be returned.
    #
    #     Returns: List of top_chunk_documents.
    #
    #     """
    #     # Encode query
    #     query_embedding = self.retriever_model.encode([self.generated_sample])
    #
    #     # Encode chunks
    #     chunk_embeddings = self.retriever_model.encode(choices)
    #
    #     # Calculate cosine similarities
    #     similarities = cosine_similarity(query_embedding, chunk_embeddings)
    #
    #     # Retrieve top-K chunks
    #     top_choice_indices = similarities.argsort(axis=1)[:, -top_k:][0]
    #     top_chunk_documents = [choices[i] for i in top_choice_indices]
    #
    #     return top_chunk_documents

    def ignore_initial_text(self):
        filtered_output = self.generated_sample.replace(self.text_to_ignore, '')
        return "### Response:" + filtered_output

    def extract_response_from_output(self, filtered_text):

        """

        r'### Response:(?:[^#]+|(?R))*### End'

        :param filtered_text:
        :return:
        """
        print("filtered_output is::::::::::::::::::::::\n\n", filtered_text)

        pattern = r'### Response:(.*?)### End'
        extracted_templates = regex.findall(pattern, filtered_text, flags=regex.DOTALL)

        # extracted_templates = re.findall(r'### Response:(.*?)###', filtered_text, re.DOTALL)
        if not extracted_templates:
            print("The list is empty")
        else:
            # print(type(extracted_templates))
            # print((extracted_templates))
            extracted_templates = " ".join(extracted_templates)
            cleaned_text = extracted_templates.replace('### Response:', '').replace('### End', '').strip()
            # print("ddd", cleaned_text)
            print("Extracted Response:\n", cleaned_text)
            return cleaned_text  # else:  #     # for temp in extracted_templates:  #     extracted_sample = " ".join(extracted_templates)  #     # split input string separated by space  #     input_uniq = Counter(extracted_sample.split(" "))  #  #     # joins two adjacent elements in iterable way  #     input_uniq = " ".join(input_uniq.keys())  #     print("Extracted Response:\n", input_uniq)  #     return input_uniq

        # print("filtered_output is::::::::::::::::::::::\n\n", filtered_text)

        # extracted_template = re.compile(r'### Response:(.*?)### End(.*?)', re.DOTALL)

        # if not extracted_template:  #     extracted_template = re.compile(r'### Response:(.*?)###', re.DOTALL)  # flags=re.IGNORECASE  # match = extracted_template.search(filtered_text)  # if match:  #     extracted_response = match.group(1).strip()  #     print("extracted_response------------------------------------>\n\n", extracted_response)  #     return extracted_response  # else:  #     print("ERORRR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", self.generated_sample)  #     return None  # raise "!!!!!!!!!!!!!!!!!No match found in the generated output {}!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(  #     filtered_text)

    def calculate_similarity(self, choices):
        # Extract response from generated output
        filtered_text = self.ignore_initial_text()

        generated_response = self.extract_response_from_output(filtered_text)
        print("Extracted answer from generated sample is:", generated_response)
        #
        # if not generated_response:
        #     return "No response extracted from the generated output."

        # Check for exact match
        if generated_response in choices:
            return generated_response  # Return the exact match
        else:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            # Encode the generated response
            print("......................................")
            generated_embedding = model.encode(generated_response, convert_to_tensor=True)

            # Encode the choices
            choices_embeddings = model.encode(choices, convert_to_tensor=True)

            # Calculate cosine similarities
            similarities = util.pytorch_cos_sim(generated_embedding, choices_embeddings)[0]

            # Find the index of the most similar choice
            similar_index = similarities.argmax().item()

            return choices[similar_index]

    def extract_choice_lists(self):
        pattern = re.compile(r'### ChoiceList:\s*(\[.*?\])', re.DOTALL)
        matches = pattern.search(self.generated_sample)

        if matches:
            choice_list_str = matches.group(1)
            return eval(choice_list_str)

        else:
            raise "!!!!!!!!!!!!!!!!!No match found in the generated output {}!!!!!!!!!!!!!!!!!!!!!!!!\n"

    def process_generated_output(self):
        choice_list = self.extract_choice_lists()
        if not choice_list:
            raise "!!!!!!!!!!!!!!!!! No choice list found in the generated output.!!!!!!!!!!!!!!!!!!!!!!\n"

        similar_choice = self.calculate_similarity(choice_list)
        # if not similar_choice:
        #     return None
        # else:
        # if not choice_list.index(similar_choice):
        #     similar_index = 2
        #     print("122121212112")
        # else:
        similar_index = choice_list.index(similar_choice)
        return similar_choice, similar_index


if __name__ == '__main__':
    xx = """
    ### Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills.
    ### Input Question:
    Everyone called him "Batman," but he knew nothing about bats and thought they were disgusting. He still cherished being referred to as Batman! How is this possible?
    and here is choice lists:
    ### ChoiceList:
    ['He tries to be friendly.', 'He is afraid others will laugh at him.', 'He was the star baseball player.', 'None of above.']
    ### Response:  He was the star baseball player.  ### End Response ### Response:  ### End Response ... (rest of your input)"""  # y = """ Instruction: As a highly proficient multiple-choice question-answering system, your task is to meticulously select the most appropriate response from the provided choice_list for the following question. Please respond after ### Response: Be cautious, as the question's context may pose a distraction. Think deeply before finalizing your choice. Pay equal attention to both correct and incorrect options to enhance your selection skills. """  #  # postprocessor = PostProcess(x, y)  #  # most_similar_choice, most_similar_index = postprocessor.process_generated_output()  # print("Most Similar Choice to the given generated:", most_similar_choice)  # print("Most Similar Index to the given generated:", most_similar_index)  # filtered_text = """  #  ### Response:  ### End.   ### Response:  #  FIVE. Remove the 2 letters F and E from five and you have in which is the Roman numeral for four.  ### End.   ### Response:  #  Take the 5 and subtract 2 from it. This results in the number 3. Now, remove the vertical top part of the digit "3" and you are left with the digit "4."   ### End.   ### Response:  #  Imagine"""  # """  # print("filtered_output is::::::::::::::::::::::\n\n", filtered_text)  #  # # Use a regular expression to find all occurrences of the pattern between '### Response:' and '### End'  #  # extracted_template = re.findall(r'### Response:(.*?)### End', filtered_text, re.DOTALL)  # if not extracted_template:  #     print("The list is empty")  # else:  #     for sample in extracted_template:  #         if not sample:  #             print("Selected sample Null", sample)  #             break  #         else:  #             extracted_response = sample.strip()  #             print("Extracted Response:\n", extracted_response)  #  #  # if extracted_templates:  #     # Extracted response is the first match  #     if len(extracted_templates) > 1:  #         if extracted_templates[0]:  #             extracted_response = extracted_templates[0].strip()  #         else:  #             extracted_templates = extracted_templates[1].strip()  #     else:  #         extracted_response = extracted_templates[0].strip()  #  # print("Extracted Response:\n", extracted_response)  # else:  #     print("ERROR: No match found.")  #     y = """
#      ### Response: ### Response:
#  Chocolate.  ### End.  ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End
# Extracted Response:
#     """
#     x = """
#      ### Response: ### Response:
#  See OH DREfford  ### End.  ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End.   ### End
#     """  # extracted_templates = re.findall(r'### Response:(.*?)### End', xx, re.DOTALL)  #  # # If no matches are found, try the pattern without '### End'  # if not extracted_templates:  #     extracted_templates = re.findall(r'### Response:(.*?)###', xx, re.DOTALL)  #  # # Strip whitespace from the extracted content  # extracted_templates = [template.strip() for template in extracted_templates]  #  # import regex  # pattern = r'### Response:(?:[^#]+|(?R))*### End'  # extracted_templates = regex.findall(pattern, filtered_text, flags=regex.DOTALL)  # print(type(extracted_templates))  # print((extracted_templates))  # xxx= " ".join(extracted_templates)  # cleaned_text = xxx.replace('### Response:', '').replace('### End', '').strip()  # print("ddd",cleaned_text)  #
