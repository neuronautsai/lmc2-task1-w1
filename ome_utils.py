"""OME-XML helper shared by inference.py and submission_check.py (no torch import)."""
import re


def ome_as_uint8(ome):
    """Copy of an OME-XML string with the Pixels type set to uint8 (the data type written by inference.py)."""
    ome = re.sub(r'(<Pixels\b[^>]*?\bType=")[^"]*(")', r"\g<1>uint8\g<2>", ome, count=1)
    return re.sub(r'(<Pixels\b[^>]*?\bSignificantBits=")[^"]*(")', r"\g<1>8\g<2>", ome, count=1)
