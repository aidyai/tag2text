# TAG GENERATOR FROM DESCRIPTIONS AND TAG INDEX PROCESSOR

# CODE FOR GENERATION

# Check if 'CAPTION_v2' column exists, if not, create it
if 'TAGS' not in df.columns:
    df['TAGS'] = pd.NA  # Initialize with NaN values

# Function to generate synonyms
def generate_synonym(caption):
    # Define input message
    messages = [
        {"role": "system", "content": "You are a highly efficient fashion tag extractor. Your task is to identify and extract fashion-related tags from any sentence provided. Focus on keywords related to clothing, accessories, styles, fabrics, colors, patterns, and fashion trends. Ensure that each tag is separated by a comma and reflects core elements of the sentence relevant to fashion. Just stick to the sentence and extract out related tags."},
        {"role": "user", "content": f"Generate the Tags for the sentence sentence : {caption}"}
    ]
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Iterate over the DataFrame rows
for index, row in df.iterrows():
    if pd.notna(row['TAGS']):
        # Skip rows where the synonym has already been generated
        continue

    # Generate synonym for the current caption
    synonym = generate_synonym(row['CAPTION'])

    # Print the original caption and its synonym
    print(f"Original: {row['CAPTION']}")
    print(f"Synonym: {synonym}")

    # Update the DataFrame
    df.at[index, 'TAGS'] = synonym

    # Save the updated row back to the CSV file
    df.to_csv('chroma_full.csv', index=False)

# Save the final DataFrame
df.to_csv('chroma_full.csv', index=False)


# TAG GENERATOR FROM DESCRIPTIONS AND TAG INDEX PROCESSOR

