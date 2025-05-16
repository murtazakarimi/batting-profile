import time

def get_key_from_value(my_dict, target_value):
    """
    Returns a list of keys from a dictionary that correspond to a given value.

    Args:
        my_dict (dict): The dictionary to search.
        target_value: The value to find the key(s) for.

    Returns:
        list: A list of keys that have the target value. Returns an empty list if the value is not found.
    """
    for key, value in my_dict.items():
        if value == target_value:
            return key

def convert_24_to_12(time_24h):
  """Converts a 24-hour time string to 12-hour format with AM/PM.

  Args:
    time_24h: A string representing time in 24-hour format (e.g., "14:30").

  Returns:
    A string representing time in 12-hour format with AM/PM (e.g., "02:30 PM").
    Returns None if the input string is not in the correct format.
  """
  try:
    time_object = time.strptime(time_24h, "%H:%M")
    time_12h = time.strftime("%I:%M %p", time_object)
    return time_12h
  except ValueError:
    return None

def parse_roster(roster_str):
    roster = {}
    lines = roster_str.strip().split('\n')
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        number = int(parts[0].lstrip("#"))
        position = parts[1]
        name = " ".join(parts[2:])
        roster[idx] = {"number": number, "position": position, "name": name}
    return roster