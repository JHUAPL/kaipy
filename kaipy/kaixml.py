# Standard modules
import os
import json
import datetime

# Third-party modules
import xml.etree.ElementTree as ET

def getXmlElement(E, name, caseInsensitive=True):
    if E is None:
        return None
    for element in E.iter():
        if caseInsensitive:
            if element.tag.casefold() == name.casefold():
                return element
        else:
            if element.tag == name:
                return element
    return None

def getXmlAttribute(E, name, caseInsensitive=True):
    if E is None:
        return None
    for attr_name,attr_value in E.attrib.items():
        if caseInsensitive:
            if attr_name.casefold() == name.casefold():
                return attr_value
        else:
            if attr_name == name:
                return attr_value
    return None
