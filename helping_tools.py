import os
import os.path


def find_named_folders(rootPath, namedFolder):
    """Searches for all subdirectories which are called <namedFolder> and retruns the absolute path.
    Useful for finding all frame-directories.
    rootPath ... string path from where to start the search into all subfolders
    namedFolder ... string name of folder for which to look out
    returns list[string] of all found absolute paths to the folders which are called <namedFolder>
    """
    folderPaths = []
    print("Looking for folder paths...")
    print("Working... (this might take a while, you can go and fetch a cup of tea)")
    for path, folders, files in os.walk(rootPath): # os.walk function requires the other variables
    # found no function to just iterate over directories
        for name in folders:
            if namedFolder in name:
                folderPaths.append(os.path.join(path, name))
    print("Found %s folder paths!" % len(folderPaths))
    return folderPaths