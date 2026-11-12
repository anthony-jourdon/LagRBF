import os, io, tempfile, shutil

PROGNAME = "LagRBF"
LANGUAGES = {
  ".c": "c",
  ".h": "c",
  ".py": "python",
}
MARKERS = {
  "c": ["/*@", "@*/"],
  "python": ['"""@', '@"""'],
}

def get_header(filename:str, language:str="c") -> str:
  s = MARKERS[language][0] + "\n"
  s += '====================================================================================================\n'
  s += '  Copyright (c) 2026\n'
  s += '      Anthony Jourdon jourdon@geoazur.unice.fr\n'
  s += '      Université Côte d\'Azur\n'
  s += '      Observatoire de la Côte d\'Azur, CNRS, IRD, Géoazur\n'
  s += '      250 Rue Albert Einstein\n'
  s += '      06560 Valbonne\n' 
  s += '      France\n'
  s += '====================================================================================================\n\n'
  s += f'  project: {PROGNAME}\n'
  s += f'  file:    {filename}\n\n'
  s += f'  This file is part of {PROGNAME} software.\n'
  s += f'  {PROGNAME} is free software: you can redistribute it and/or modify it under the terms of the GNU\n'
  s +=  '  General Public License as published by the Free Software Foundation, either version 3 of the\n'
  s +=  '  License, or any later version.\n\n'
  s += f'  {PROGNAME} is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;\n'
  s +=  '  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n'
  s +=  '  See the GNU General Public License for more details.\n\n'
  s += f'  You should have received a copy of the GNU General Public License along with {PROGNAME}.\n'
  s +=  '  If not, see <https://www.gnu.org/licenses/>.\n'
  s += MARKERS[language][1] + "\n\n"
  return s

def skip_header(src:io.TextIOWrapper, language:str) -> None:
  """Advance src past an existing header (markers included) and one trailing blank line."""
  # get opening and closing markers for the given language
  start, end = MARKERS[language]
  # register the current position in the file
  pos = src.tell()
  # read the first line and check if it starts with the opening marker
  if not src.readline().startswith(start):
    src.seek(pos) # if not, reset the file position and return
    return
  # read lines of the header until the closing marker is found
  while True:
    line = src.readline()
    if line == "":
      raise ValueError("closing marker not found")
    if line.startswith(end):
      break
  # after the closing marker, check for a blank line and skip it if present
  pos = src.tell()
  if src.readline().strip() != "":   # swallow one blank separator line only
    src.seek(pos)
  return

def prepend_header(root:str, fname:str, language:str, reset:bool):
  path = os.path.join(root, fname)
  with open(path, "r", encoding="utf-8", newline="") as src:
    # check if the file already has a licence header
    has_header = src.readline().startswith(MARKERS[language][0])
    src.seek(0)
    # if the file has a header and reset is False, skip it
    if has_header and not reset:
      print(f"Licence header already exists in {fname}, skipping...")
      return
    # if reset is True, remove the existing header by 
    # setting the file position to the start of the header and skipping it
    if has_header:
      print(f"Resetting licence header in {fname}")
      skip_header(src, language)      # raises before the temp file exists

    # create a temporary file in the same directory to write the new content
    with tempfile.NamedTemporaryFile(
      "w", encoding="utf-8", newline="",
      dir=root, delete=False
    ) as tmp:
      try:
        # write the new header to the temporary file
        tmp.write(get_header(fname, language))
        # copy the rest of the original file to the temporary file
        shutil.copyfileobj(src, tmp, 1 << 16)
      except BaseException:
        os.unlink(tmp.name)
        raise
  # replace the original file with the temporary file, preserving permissions
  shutil.copymode(path, tmp.name)
  os.replace(tmp.name, path)

def search_and_prepend(directory:str, reset:bool=False):
  for root, dirs, files in os.walk(directory):
    for file in files:
      extension = os.path.splitext(file)[1]
      if extension in LANGUAGES:
        print(f"Processing file: {file} in directory: {root}")
        try:
          prepend_header(root, file, LANGUAGES[extension], reset=reset)
        except Exception as e:
          print(f"Error processing file: {file} in directory: {root}, error: {e}")
    # recursively process subdirectories
    for dir in dirs:
      search_and_prepend(dir, reset=reset)
  return

if __name__ == "__main__":
  
  import argparse

  parser = argparse.ArgumentParser(description="Add licence header to a source file")
  parser.add_argument("-d", "--directory", type=str, default=".", help="Directory to search for source files")
  parser.add_argument("-r", "--reset", action="store_true", help="Reset the licence header if it already exists", default=False)
  args = parser.parse_args()
  search_and_prepend(args.directory, reset=args.reset)
