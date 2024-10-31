#   python3 convert_labels_to_single_lego_label.py
# step 2: let all the labels in the xml files be 'lego'
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def update_xml_labels(folder_path, new_label='lego', verbose=True):
    """
    Update all XML annotation files in the specified folder to use a single label.
    
    Args:
        folder_path (str): Path to the folder containing XML files
        new_label (str): The new label to use (default: 'lego')
        verbose (bool): Whether to print detailed progress messages
    
    Returns:
        tuple: (number of files processed, number of files updated)
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")
    
    files_processed = 0
    files_updated = 0
    
    # Process all XML files in the folder
    for xml_path in folder.glob('*.xml'):
        try:
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Track if this file needs updates
            file_updated = False
            
            # Update all object names to the new label
            for obj in root.findall('.//object/name'):
                if obj.text != new_label:
                    obj.text = new_label
                    file_updated = True
            
            # Save the file if changes were made
            if file_updated:
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                files_updated += 1
                if verbose:
                    print(f"Updated labels in: {xml_path.name}")
            
            files_processed += 1
            
        except ET.ParseError:
            print(f"Error: Could not parse XML file: {xml_path.name}")
        except Exception as e:
            print(f"Error processing {xml_path.name}: {str(e)}")
    
    return files_processed, files_updated

def main():
    # Define the folders to process
    folders = [
        '/Users/sophiawang/Desktop/Lab3/training set350',
        '/Users/sophiawang/Desktop/Lab3/Validation Set75',
        '/Users/sophiawang/Desktop/Lab3/Testing Set75'
    ]
    
    total_processed = 0
    total_updated = 0
    
    # Process each folder
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        try:
            processed, updated = update_xml_labels(folder)
            total_processed += processed
            total_updated += updated
            print(f"Files processed: {processed}")
            print(f"Files updated: {updated}")
        except Exception as e:
            print(f"Error processing folder {folder}: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {total_processed}")
    print(f"Total files updated: {total_updated}")

if __name__ == "__main__":
    main()