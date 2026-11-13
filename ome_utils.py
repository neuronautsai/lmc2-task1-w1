"""OME-XML helpers shared by inference.py and submission_check.py (no torch import)."""
import re

# output mode -> (numpy dtype name, OME-XML pixel type, significant bits)
OUTPUT_TYPES = {"uint8": ("uint8", "uint8", 8), "float32": ("float32", "float", 32)}


def ome_with_type(ome, ome_type, bits):
    """Copy of an OME-XML string with the Pixels type and SignificantBits set to the written data type."""
    ome = re.sub(r'(<Pixels\b[^>]*?\bType=")[^"]*(")', rf"\g<1>{ome_type}\g<2>", ome, count=1)
    return re.sub(r'(<Pixels\b[^>]*?\bSignificantBits=")[^"]*(")', rf"\g<1>{bits}\g<2>", ome, count=1)


def ome_as_uint8(ome):
    return ome_with_type(ome, "uint8", 8)
