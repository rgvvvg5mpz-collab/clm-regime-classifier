"""The 7-way regime label space and the text CLM sees for each class.

A message (client message or assistant response from the firm's GenAI channel) is
assigned to the rule regime it implicates, or ``compliant`` when it implicates none.
"""

INSTRUCTIONS = ("Which securities regulation does this message from a broker-dealer's "
                "AI chat channel potentially violate or trigger, if any?")

# Candidate texts for the action head (plain answer prose, as the heads were trained on).
CLASSES: dict[str, str] = {
    "compliant": "None. The message is compliant: an ordinary question, general education, "
                 "or a properly balanced and disclosed answer that raises no regulatory issue.",
    "finra_2210": "FINRA Rule 2210 communications with the public: the assistant's response is "
                  "misleading, promissory, exaggerated, unbalanced on risks and benefits, projects "
                  "performance, mischaracterizes taxes, or omits required disclosures.",
    "reg_bi": "Regulation Best Interest: the assistant makes a personalized recommendation of a "
              "security, investment strategy, account type or rollover, or steers the customer "
              "toward proprietary products.",
    "finra_4530": "FINRA Rule 4530 customer complaint: the customer is making a written grievance "
                  "about sales practices, unauthorized trading, account errors, theft, "
                  "misappropriation or forgery that must be reported.",
    "sec_17a3_17a4": "SEC Rules 17a-3 and 17a-4 books and records: the customer references or "
                     "invites off-channel communication (texts, personal phone, personal email), asks "
                     "that something not be recorded or written down, or reports an account update "
                     "that must be captured in the firm's records.",
    "reg_sp": "Regulation S-P privacy: the customer shares sensitive personal information such as an "
              "SSN, date of birth or account numbers, makes a privacy or opt-out request, reports a "
              "data breach or incident, or asks for another person's information.",
    "reg_sid": "Regulation S-ID identity theft red flags: the message shows signs of identity theft "
               "or account takeover, such as a reported stolen identity, inconsistent or suspicious "
               "identifying information, unauthorized contact-detail changes, or attempts to bypass "
               "or delegate identity verification.",
}
LABELS = list(CLASSES)
