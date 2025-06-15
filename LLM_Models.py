from openai import OpenAI
import json
from credentials import openai_key, deepseek_key


def DeepSeek(transcription: str) -> dict:
    """
    Extract structured information from a call transcription using the DeepSeek API.

    Returns default empty values if transcription is too short to analyze.
    """
    # If the transcription is trivial, return defaults without calling the API
    if len(transcription) < 10:
        return {
            'Conversation': [],
            'Doctor Name': "N/A",
            'Patient Name': "N/A",
            'Email': "N/A",
            'Sentiment': 'Neutral',
            'Summery': "N/A"
        }

    # Initialize DeepSeek client with API key and base URL
    client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    # Send the transcription to the DeepSeek chat completion endpoint
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '''
    You are a specialized assistant focused on extracting critical information from dental lab call transcriptions. Your primary task is to accurately identify and extract the patient name, doctor name, and email address (if available). You must carefully analyze the entire conversation to ensure the most accurate data extraction.

    ### CRITICAL EXTRACTION REQUIREMENTS:
    1. **Patient Name (HIGHEST PRIORITY)**: 
    - Extract FULL patient name (first and last)
    - Pay careful attention to spelling, especially when names are spelled out letter by letter
    - Identify which is the first name and which is the last name, even if initially confused in the conversation
    - If multiple potential patient names appear, look for context clues to determine the actual patient

    2. **Doctor Name (HIGHEST PRIORITY)**:
    - Extract the complete doctor name with title (Dr., Doctor, etc.)
    - Look for any context clues that confirm someone is the treating dentist/doctor
    - Be careful not to misidentify staff or lab personnel as doctors

    3. **Email Address**:
    - Extract any email addresses mentioned in the conversation
    - If no email is mentioned, use "N/A"

    ### CONVERSATION ANALYSIS:
    - Break down the full conversation into speaker turns
    - Identify speakers as specifically as possible (by name if given, or by role)
    - Maintain the exact flow and content of the conversation

    ### SENTIMENT ANALYSIS:
    - Classify the overall sentiment as one of the following:
    - Negative: Clear frustration, complaints, or dissatisfaction expressed
    - Neutral: Professional, matter-of-fact conversation without strong emotions
    - Positive: Clear expression of satisfaction, gratitude, or friendly rapport

    ### OUTPUT FORMAT:
    Provide a JSON object with the following structure:
    ```json
    {
        "Conversation": [
            {"speaker": "[Speaker Name/Role]", "message": "[Exact message content]"},
            ...
        ],
        "Doctor Name": "[Full doctor name with title]",
        "Patient Name": "[Full patient name - First Last]",
        "Email": "[Email address or N/A]",
        "Sentiment": "[Negative/Neutral/Positive]",
        "Summary": "[Brief summary of the conversation purpose and outcome]"
    }
    ```

    ### EXAMPLES:

    Example 1:
    Transcription: "Good morning, I'm Marit from Clear View Dental Lab. I need to confirm details about Dr. Peterson's patient, Sarah Johnson. Her crown is ready but I need to verify her appointment date." 

    Output:
    ```json
    {
        "Conversation": [
            {"speaker": "Marit", "message": "Good morning, I'm Marit from Clear View Dental Lab. I need to confirm details about Dr. Peterson's patient, Sarah Johnson. Her crown is ready but I need to verify her appointment date."}
        ],
        "Doctor Name": "Dr. Peterson",
        "Patient Name": "Sarah Johnson",
        "Email": "N/A",
        "Sentiment": "Neutral",
        "Summary": "Marit from Clear View Dental Lab called to confirm appointment details for Dr. Peterson's patient Sarah Johnson, whose crown is ready."
    }
    ```

    Example 2 (Challenging name identification):
    Transcription: "Hi, this is Jennifer from Bay Area Dental. I'm calling about patient Martinez, Carlos. Dr. Wang requested some changes to his partial denture. The patient's name might be in our system as C. Martinez or Carlos M."

    Output:
    ```json
    {
        "Conversation": [
            {"speaker": "Jennifer", "message": "Hi, this is Jennifer from Bay Area Dental. I'm calling about patient Martinez, Carlos. Dr. Wang requested some changes to his partial denture. The patient's name might be in our system as C. Martinez or Carlos M."}
        ],
        "Doctor Name": "Dr. Wang",
        "Patient Name": "Carlos Martinez",
        "Email": "N/A",
        "Sentiment": "Neutral",
        "Summary": "Jennifer from Bay Area Dental called regarding Dr. Wang's request for changes to patient Carlos Martinez's partial denture."
    }
    ```
    Example 3:
    Transcription: "Good morning, I'm Marit. How are you? Good, good. I talked to Dr. Khalil about this case for me, Jada. I asked him when the patient was scheduled, but he said call the office to find out. Do you know when this was scheduled? I'm sorry, where are you from? I'm from the National Institute of Medicine and Phenia dental lab. Can I have the patient's last name? Sure, Jada. J-A-D-A. First name? First name, it looks like, Myit, M-E-I-T-E. Okay, so that's the last name. First name is Jada. Okay. She scheduled for April 14th, Monday. April 14th? No problem. How do you spell your name, Marit? The merit, D-A-M-S-M-A-R-I-S. The merit, D-A-M-A-R-T-S. The merit. I'm sorry, I think it's very well. D-A-M-S-M-A-R-T-S? Yes. Okay, thank you very much. You're welcome. Okay, bye-bye."
            
    Output:
    ```json
    {
        "Conversation": [
        {"speaker": "Marit", "message": "Good morning, I'm Marit. How are you?"},
        {"speaker": "Office staff/representative", "message": "Good, good."},
        {"speaker": "Marit", "message": "I talked to Dr. Khalil about this case for me, Jada. I asked him when the patient was scheduled, but he said call the office to find out. Do you know when this was scheduled?"},
        {"speaker": "Office staff/representative", "message": "I'm sorry, where are you from?"},
        {"speaker": "Marit", "message": "I'm from the National Institute of Medicine and Phenia dental lab."},
        {"speaker": "Office staff/representative", "message": "Can I have the patient's last name?"},
        {"speaker": "Marit", "message": "Sure, Jada. J-A-D-A."},
        {"speaker": "Office staff/representative", "message": "First name?"},
        {"speaker": "Marit", "message": "First name, it looks like, Myit, M-E-I-T-E."},
        {"speaker": "Office staff/representative", "message": "Okay, so that's the last name. First name is Jada. Okay. She scheduled for April 14th, Monday."},
        {"speaker": "Marit", "message": "April 14th? No problem. How do you spell your name, Marit?"},
        {"speaker": "Office staff/representative", "message": "The merit, D-A-M-S-M-A-R-I-S. The merit, D-A-M-A-R-T-S. The merit. I'm sorry, I think it's very well. D-A-M-S-M-A-R-T-S?"},
        {"speaker": "Marit", "message": "Yes."},
        {"speaker": "Office staff/representative", "message": "Okay, thank you very much."},
        {"speaker": "Marit", "message": "You're welcome. Okay, bye-bye."}
    ],
        "Doctor Name": "Dr. Khalil",
        "Patient Name": "Jada Meite",
        "Email": "N/A",
        "Sentiment": "Neutral",
        "Summary": "Marit from the dental lab called to inquire about patient Jada Meite's appointment scheduled for April 14th. Dr. Khalil had referred them to the office for scheduling details"
    }
    ```

    If you did not find any specific field information form it then set that field value to "N/A" for example this is the transcription: "Hello how are you? Can hear me? Hello can you hear me?"
        
    Then your output:
    ```json
    {
        "Conversation": [
            {"speaker": "Unknown", "message": "Hello how are you? Can hear me? Hello can you hear me?"}
        ],
        "Doctor Name": "N/A",
        "Patient Name": "N/A",
        "Email": "N/A",
        "Sentiment": "Neutral",
        "Summary": "N/A"
    }
    ```

    Remember: Your MOST IMPORTANT task is to accurately extract the doctor name and patient name. Analyze the entire conversation carefully before determining these crucial pieces of information. Return ONLY the JSON output without any additional text or explanation.
            '''},
            {"role": "user", "content": transcription},
        ],
        stream=False
    )

    # The API returns content which should be a JSON string; handle code-block wrapping
    content = response.choices[0].message.content
    try:
        # If wrapped in ```json ... ``` remove the markers
        if content.startswith("```json"):
            json_text = content.split('```json')[1].rsplit('```', 1)[0].strip()
        else:
            json_text = content
        response_dict = json.loads(json_text)
    except Exception as e:
        # On parse error, log and return defaults
        print("Error parsing DeepSeek response:", e)
        print("Raw response content:", content)
        return {
            'Conversation': [],
            'Doctor Name': "N/A",
            'Patient Name': "N/A",
            'Email': "N/A",
            'Sentiment': 'Neutral',
            'Summery': "N/A"
        }

    return response_dict


def verify_with_gpt4(deepseek_response: dict, transcription: str) -> dict:
    """
    Verify and correct extracted fields using GPT-4. If everything is correct,
    returns the original deepseek_response; otherwise returns the corrected JSON.
    """
    try:
        # Initialize GPT-4 client with OpenAI key
        client = OpenAI(api_key=openai_key)

        # Build a detailed verification prompt including original data
        prompt = f"""
    You are an expert verification assistant specialized in reviewing dental call transcription data. Your job is to carefully verify the accuracy of the extracted information, with special focus on the Doctor Name and Patient Name fields.

    ### YOUR TASK:
    1. Review the entire conversation and transcription
    2. Verify that the extracted Doctor Name, Patient Name, and Email (if available) are accurate
    3. Check if the sentiment classification is reasonable
    4. Confirm the summary accurately reflects the conversation's key points

    ### CRITICAL VERIFICATION POINTS:
    - DOCTOR NAME: Is the full doctor name (with title) correctly extracted? Look for any mentions of doctors in the conversation.
    - PATIENT NAME: Is the patient name correctly identified with proper first/last name order? Pay special attention to any spelled-out names.
    - EMAIL: If any email address is mentioned, is it correctly extracted?

    ### OUTPUT FORMAT:
    - If all information is correctly extracted, simply return: `None`
    - If any information needs correction, provide the complete corrected JSON using EXACTLY the same format as the original

    IMPORTANT: Do not provide any explanations or additional text in your response. Return either `None` or the corrected JSON only.

    Original transcription:
    {transcription}

    Extracted data to verify:
    {json.dumps(deepseek_response, indent=2)}
"""

        # Send prompt to GPT-4.1 model, asking for either 'None' or corrected JSON
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=25000
        )
        verification = response.choices[0].message.content.strip()

        # If GPT indicates no change, return original extraction
        if verification == "None":
            return deepseek_response

        # Otherwise parse the corrected JSON
        # Remove optional markdown fencing
        if verification.startswith("```"):
            verification = verification.strip('`')
        corrected = json.loads(verification)
        print("--- GPT-4o corrections applied ---")
        return corrected

    except json.JSONDecodeError as e:
        print("Error decoding GPT-4o response:", e)
        print("Response was:", verification)
        return deepseek_response
    except Exception as e:
        # On any other error, log and fallback
        print("Error in GPT-4o verification:", e)
        return deepseek_response
