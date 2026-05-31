import collections.abc

def _deep_update(dictionary: dict, update: dict) -> dict:
    """
    Recursively updates a nested dictionary d with vals from u

    Args:
        dictionary (dict): The original dict to be updated
        update (dict): The dict with vals to update the original dict with
    
    Returns:
        dict: The updated dictionary with the same structure as the original dict but with updated vals from the update dict
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = _deep_update(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary