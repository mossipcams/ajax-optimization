
from __future__ import annotations

import random
import re
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    SATELLITE_NAME_BRAND,
    SATELLITE_NAME_SUFFIX,
    CapabilitySpec,
    required_features,
)

SATELLITE_RATE = 0.30

_AREAS = (
    "Kitchen", "Living Room", "Dining Room", "Master Bedroom", "Guest Bedroom", "Kids Room",
    "Nursery", "Bathroom", "Ensuite", "Powder Room", "Office", "Study", "Den", "Family Room",
    "Garage", "Workshop", "Shed", "Basement", "Attic", "Hallway", "Landing", "Entryway", "Foyer",
    "Mudroom", "Laundry Room", "Utility Room", "Pantry", "Closet", "Porch", "Patio", "Deck",
    "Balcony", "Garden", "Backyard", "Front Yard", "Driveway", "Sunroom", "Playroom",
    "Guest Suite", "Spare Room", "Home Gym", "Music Room", "Craft Room", "Storage Room",
    "Reading Nook", "Games Room", "Boiler Room", "Server Room", "Dressing Room", "Cloakroom",
    "Scullery", "Bonus Room", "Teen Room", "Front Bedroom", "Back Bedroom", "Staircase",
    "Sewing Room", "Art Room", "Yoga Room", "Toy Room", "Linen Closet", "Granny Flat",
    "Guest House", "Barn", "Pool House", "Carport",
    "Gazebo", "Pool Deck", "Side Yard", "Rooftop", "Pergola", "Lanai", "Fire Pit", "Back Porch",
    "Vegetable Patch",
)
_ROLES = {
    "lights": (
        "Ceiling Light", "Wall Light", "Lamp", "Ceiling Lights", "Spotlights", "Pendant Light",
        "Downlights", "Recessed Lights", "Track Lights", "Flush Mount Light", "Globe Light",
        "Bulb", "Arc Lamp", "Lantern", "Night Light", "Cove Lights", "Panel Light",
    ),
    "fans": (
        "Ceiling Fan", "Floor Fan", "Desk Fan", "Tower Fan", "Box Fan", "Window Fan", "Wall Fan",
        "Air Circulator", "Clip Fan", "Table Fan",
    ),
    "switches": (
        "Plug", "Outlet", "Lamp Plug", "Charger Plug", "Heater Plug", "Aquarium Plug",
        "Printer Plug", "Router Plug", "Monitor Plug", "Radio Plug", "Fountain Plug",
        "Wall Outlet",
    ),
    "covers": (
        "Window Blinds", "Left Window Blinds", "Right Window Blinds", "Roller Blinds",
        "Blackout Blinds", "Skylight Blinds", "Glass Door Blinds", "Vertical Blinds",
        "Bay Window Blinds", "Pleated Blinds", "Honeycomb Blinds", "Sheer Blinds",
    ),
    "locks": (
        "Front Door Lock", "Back Door Lock", "Side Door Lock", "Entry Door Lock", "Pet Door Lock",
        "French Door Lock", "Rear Door Lock", "Gate Lock", "Utility Door Lock",
        "Sliding Door Lock", "Storm Door Lock",
    ),
    "media_players": (
        "TV", "Television", "OLED TV", "Streaming TV", "Speaker", "Soundbar", "Smart Speaker",
        "Echo Dot", "QLED TV", "Smart TV", "Apple TV", "Chromecast", "Fire TV", "Shield TV",
        "Nest Mini", "HomePod Mini", "Sonos Speaker", "Bookshelf Speakers", "Subwoofer", "Radio",
        "Echo Show", "Streaming Stick",
    ),
    "climate": (
        "Thermostat", "Heating", "Heat Pump", "Air Conditioner", "Mini Split", "Boiler", "AC Unit",
        "Furnace", "Window AC", "Wall Heater", "Ductless AC",
    ),
    "vacuums": (
        "Robot Vacuum", "Vacuum", "Robot Cleaner", "Floor Cleaner", "Cleaning Robot",
        "Vacuum Robot", "Robovac", "Hoover", "Floor Robot",
    ),
    "scripts": (
        "Good Morning", "Good Night", "Leaving Home", "Welcome Home", "Movie Time", "Bedtime",
        "All Lights Off", "Water the Plants", "Feed the Cat", "Evening Cleanup", "Lock the House",
        "Morning Coffee", "School Run", "Bath Time", "Story Time", "Game Night", "Vacation Mode",
        "Guest Mode", "Feed the Dog", "Walk the Dog", "Heat the House", "Wind Down", "Quiet Hours",
        "Lights Out", "Piano Practice",
    ),
    "scenes": (
        "Movie Night", "Dinner Time", "Reading Time", "Relax", "Night Lights", "Bedtime Lights",
        "Game Night", "Cozy Lights", "Late Night", "Warm Glow", "Focus Lights", "Soft Lights",
        "Wind Down", "Daylight", "Snack Time",
    ),
    "lawn_mowers": (
        "Robot Mower", "Lawn Mower", "Mower", "Robotic Mower", "Grass Cutter", "Lawn Robot",
        "Mowing Robot",
    ),
    "todo_lists": (
        "Groceries", "Shopping List", "Household Supplies", "Weekend Chores", "Hardware Store",
        "Packing List", "Gift Ideas", "Pet Supplies", "School Supplies", "Home Repairs",
        "Meal Plan", "Errands", "Lawn Care", "Baby Supplies", "Car Maintenance",
    ),
    "buttons": (
        "Wall Button", "Light Button", "Remote Button", "Bedside Button",
        "Desk Button", "Mode Button", "Shortcut Button", "Push Button",
    ),
}
_ROOM_LIGHTS = {
    "Kitchen": (
        "Ceiling Lights", "Counter Lights", "Under Cabinet Lights", "Island Pendants",
        "Sink Light", "Downlights", "Pendant Lights", "Kickboard Lights", "Hood Light",
        "Recessed Lights", "Plinth Lights", "Track Lights",
    ),
    "Dining Room": (
        "Chandelier", "Pendant Lights", "Wall Lights", "Table Lamp", "Buffet Lamp",
        "Sideboard Lamp", "Candle Bulbs", "Downlights", "Picture Light", "Ceiling Light",
        "Linear Pendant",
    ),
    "Bathroom": (
        "Mirror Light", "Vanity Lights", "Ceiling Light", "Shower Light", "Downlights",
        "Bath Light", "Night Light", "Medicine Cabinet Light", "Tub Light", "Heat Lamp",
    ),
    "Bedroom": (
        "Bedside Lamp", "Reading Lamp", "Ceiling Light", "Left Bedside Lamp", "Right Bedside Lamp",
        "Headboard Light", "Dresser Lamp", "Closet Light", "Wardrobe Light", "Night Light",
        "Pendant Light", "Fairy Lights", "Salt Lamp",
    ),
    "Living Room": (
        "Floor Lamp", "Table Lamp", "TV Backlight", "Ceiling Lights", "Reading Lamp", "Arc Lamp",
        "Picture Light", "Downlights", "Bookcase Lights", "Fireplace Lights", "Couch Lamp",
        "Wall Lights", "Lantern",
    ),
    "Office": (
        "Desk Lamp", "Monitor Backlight", "Ceiling Light", "Reading Lamp", "Bookcase Lights",
        "Panel Light", "Downlights", "Key Light", "Ring Light", "Banker Lamp", "Wall Light",
    ),
    "Study": (
        "Desk Lamp", "Reading Lamp", "Ceiling Light", "Bookcase Lights", "Banker Lamp",
        "Wall Light", "Downlights", "Monitor Backlight",
    ),
    "Hallway": (
        "Ceiling Light", "Downlights", "Night Light", "Stair Lights", "Wall Lights",
        "Pendant Light", "Motion Light", "Step Lights",
    ),
    "Garage": (
        "Ceiling Light", "Shop Lights", "Workbench Light", "Batten Light", "Floodlight",
        "Panel Light", "Tube Lights",
    ),
    "Outside": (
        "Wall Light", "Wall Lights", "Floodlight", "Path Lights", "String Lights", "Lanterns",
        "Step Lights", "Bollard Lights", "Security Light", "Festoon Lights", "Spike Lights",
        "Post Light",
    ),
}
_BRANDS = {
    "lights": (
        "Hue", "Ikea", "LIFX", "Nanoleaf", "Govee", "Lutron", "WiZ", "Sengled", "Cync", "Innr",
        "Ledvance", "Tapo", "Twinkly",
    ),
    "fans": (
        "Dyson", "Honeywell", "Vornado", "Bond", "Haiku", "Modern Forms", "Xiaomi", "Dreo",
        "SwitchBot", "Tuya",
    ),
    "switches": (
        "Shelly", "Kasa", "Wyze", "Sonoff", "Tapo", "Zooz", "Inovelli", "Leviton", "Aqara",
        "TP-Link", "Tuya",
    ),
    "covers": (
        "Ikea", "Somfy", "SwitchBot", "Lutron", "Aqara", "Hunter Douglas", "PowerView",
        "Smartwings", "Soma", "Zemismart",
    ),
    "locks": (
        "Yale", "August", "Schlage", "Aqara", "Kwikset", "Ultraloq", "Lockly", "Level",
        "SwitchBot", "Eufy",
    ),
    "media_players": (
        "Samsung", "LG", "Sony", "Roku", "Vizio", "Panasonic", "TCL", "Hisense", "Sharp",
        "Toshiba", "Insignia",
    ),
    "climate": (
        "Nest", "Ecobee", "Honeywell", "Tado", "Sensibo", "Mysa", "Netatmo", "Hive", "Sensi",
        "Venstar",
    ),
    "vacuums": (
        "Roborock", "Roomba", "Deebot", "Eufy", "Ecovacs", "Neato", "Narwal", "Xiaomi",
        "SwitchBot", "Tapo",
    ),
    "lawn_mowers": (
        "Husqvarna", "Worx", "Segway", "Gardena", "Mammotion", "Bosch", "Kress", "Landxcape",
        "Stihl", "Honda",
    ),
    "buttons": (
        "Flic", "Aqara", "Hue", "Ikea", "Shelly", "SwitchBot", "Lutron", "Sonoff", "Tuya",
        "Third Reality",
    ),
}
_FIRST_NAMES = (
    "Joe", "Priya", "Wei", "Amara", "Sofia", "Yusuf", "Ingrid", "Mateo", "Rosa", "Dmitri",
    "Fatima", "Kenji", "Niamh", "Tomas", "Leila", "Bram", "Oona", "Hallie", "Noor", "Ellis",
    "Hana", "Ivy", "Dev", "Leona", "Oscar", "Wren", "Kwame", "Ama", "Kofi", "Abena", "Chidi",
    "Adaeze", "Emeka", "Folake", "Tunde", "Yetunde", "Sipho", "Thandi", "Zanele", "Lindiwe",
    "Bongani", "Amani", "Jabari", "Zuri", "Wanjiru", "Kamau", "Nia", "Oluwaseun", "Ayodele",
    "Kehinde", "Mandla", "Nomvula", "Dayo", "Arjun", "Ananya", "Rohan", "Kavya", "Vikram", "Meera",
    "Aarav", "Ishaan", "Diya", "Sanjay", "Lakshmi", "Aditi", "Nikhil", "Pooja", "Rahul", "Sunita",
    "Anil", "Harpreet", "Gurpreet", "Imran", "Ayesha", "Zainab", "Bilal", "Farhan", "Nadia",
    "Saanvi", "Kiran", "Mei", "Jun", "Hiroshi", "Yuki", "Sakura", "Haruto", "Aiko", "Ren", "Jiwoo",
    "Hyun", "Jisoo", "Minho", "Xiu", "Ming", "Lan", "Bao", "Linh", "Minh", "Thao", "Duc", "Hoa",
    "Akira", "Kaito", "Haruka", "Daisuke", "Emi", "Takumi", "Jia", "Sora", "Riku", "Omar", "Layla",
    "Hassan", "Rania", "Karim", "Yasmin", "Samir", "Dalia", "Mona", "Ziad", "Hiba", "Nabil",
    "Salma", "Amir", "Soraya", "Dariush", "Shirin", "Reza", "Parisa", "Cyrus", "Mehmet", "Ayse",
    "Emre", "Zeynep", "Deniz", "Burak", "Lars", "Astrid", "Freya", "Sven", "Bjorn", "Sigrid",
    "Henrik", "Elin", "Mikkel", "Signe", "Aino", "Eero", "Pieter", "Anouk", "Joost", "Lotte",
    "Matthias", "Greta", "Lukas", "Anja", "Klaus", "Heidi", "Pierre", "Camille", "Margaux",
    "Lucien", "Elodie", "Giulia", "Luca", "Marco", "Chiara", "Alessandro", "Francesca", "Lorenzo",
    "Carmen", "Javier", "Lucia", "Pablo", "Joao", "Beatriz", "Tiago", "Agnieszka", "Piotr",
    "Kasia", "Marek", "Zofia", "Ondrej", "Petra", "Milan", "Ivana", "Nikola", "Dragan",
    "Anastasia", "Sergei", "Olga", "Natasha", "Yuri", "Katya", "Bogdan", "Oksana", "Taras",
    "Eleni", "Nikos", "Yiannis", "Cormac", "Siobhan", "Aoife", "Ciaran", "Declan", "Eilidh",
    "Morag", "Hamish", "Rhys", "Cerys", "Dylan", "Bethan", "Gethin", "Diego", "Valentina",
    "Camila", "Santiago", "Ximena", "Joaquin", "Paloma", "Rafael", "Guadalupe", "Alejandro",
    "Renata", "Thiago", "Luana", "Caio", "Oliver", "Harriet", "Felix", "Imogen", "Theo", "Poppy",
    "Arthur", "Florence", "Jasper", "Matilda", "Rupert", "Beatrix", "Hazel", "Miles", "Nora",
    "Silas", "Clara", "Ezra", "Iris", "Jude", "Maeve", "Rory", "Tessa", "Wyatt", "Hattie", "Otis",
    "Ada", "Bea", "Elsie", "Archie", "Aroha", "Tane", "Mere", "Wiremu", "Keanu", "Leilani",
    "Malia", "Kai", "Alice", "Ben", "Chloe", "David", "Emma", "Farid", "Grace", "Hugo", "Ines",
    "Jonas", "Adam", "Abigail", "Aidan", "Akosua", "Alan", "Alicia", "Amelia", "Andre", "Andrea",
    "Angela", "Anika", "Anton", "Ariana", "Arnav", "Asha", "Aurora", "Ayumi", "Basil", "Benedikt",
    "Bianca", "Brendan", "Bruno", "Callum", "Carla", "Caroline", "Cecilia", "Chiamaka",
    "Christina", "Cian", "Claudia", "Colin", "Connor", "Daniela", "Daniel", "Darius", "Delia",
    "Dennis", "Dina", "Dominic", "Edgar", "Edith", "Eduardo", "Elena", "Elias", "Eliza", "Ella",
    "Emil", "Eric", "Esther", "Ethan", "Eva", "Fabian", "Fiona", "Francis", "Frida", "Gabriel",
    "Gemma", "George", "Gloria", "Gustav", "Hamza", "Hannah", "Harold", "Helen", "Ibrahim",
    "Irene", "Isaac", "Isabel", "Ivan", "Jack", "Jacob", "Jade", "James", "Jana", "Jasmine", "Jay",
    "Jennifer", "Joel", "Jordan", "Jose", "Joseph", "Julia", "Julian", "June", "Karen", "Karl",
    "Kate", "Kayode", "Keith", "Kevin", "Laura", "Leo", "Liam", "Lily", "Lina", "Logan", "Louise",
    "Lucas", "Lydia", "Madeleine", "Magnus", "Maria", "Mark", "Martin", "Maya", "Megan", "Michael",
    "Mila", "Mohammed", "Monica", "Nathan", "Neil", "Nicole", "Noah", "Olivia", "Owen", "Patrick",
    "Paula", "Paul", "Peter", "Rachel", "Rebecca", "Ricardo", "Robert", "Ruth", "Ryan", "Sara",
    "Sean", "Sebastian", "Simon", "Sophie", "Stefan", "Stella", "Stephen", "Susan", "Teresa",
    "Thomas", "Tim", "Tobias", "Uma", "Ursula", "Valerie", "Vanessa", "Victor", "Vivian", "Walter",
    "Wendy", "William", "Xavier", "Yara", "Yvonne", "Zachary", "Zara", "Zoe", "Abdul", "Abdullah",
    "Abel", "Abraham", "Adeline", "Adrian", "Adrienne", "Agatha", "Ahmed", "Aisha", "Albert",
    "Alberto", "Alessia", "Alfred", "Alina", "Alistair", "Althea", "Amelie", "Amos", "Amrita",
    "Anais", "Anders", "Angel", "Ansel", "Anushka", "Antonio", "April", "Ari", "Ariel", "Arlo",
    "Arman", "Armando", "Asher", "Asma", "Athena", "Augustin", "Austin", "Ava", "Axel", "Aylin",
    "Aziza", "Barbara", "Beatrice", "Benedict", "Bernadette", "Bernard", "Beth", "Betty",
    "Beverly", "Blake", "Blythe", "Bonnie", "Boris", "Brandon", "Brenda", "Brian", "Bridget",
    "Brittany", "Brody", "Caleb", "Calvin", "Cassandra", "Catherine", "Cedric", "Celeste", "Celia",
    "Charles", "Charlotte", "Chester", "Christian", "Christopher", "Clarence", "Clement",
    "Clementine", "Cora", "Corinne", "Cosima", "Curtis", "Cynthia", "Daisy", "Dakota", "Damian",
    "Daphne", "Deborah", "Desmond", "Diana", "Dolores", "Donovan", "Dorothy", "Douglas", "Duncan",
    "Eamon", "Eden", "Edmund", "Edward", "Edwin", "Eileen", "Eleanor", "Elisa", "Elizabeth",
    "Eloise", "Emanuel", "Emilia", "Emily", "Enid", "Enzo", "Ernest", "Esme", "Estelle", "Eugene",
    "Evelyn", "Everett", "Faisal", "Faye", "Fern", "Finlay", "Finn", "Flora", "Frances",
    "Frederick", "Gabriela", "Gareth", "Gavin", "Georgina", "Gerald", "Geraldine", "Gideon",
    "Gilbert", "Giovanna", "Giselle", "Gordon", "Gregory", "Gwen", "Hadley", "Hakeem", "Hanan",
    "Harlow", "Harry", "Harvey", "Hector", "Helena", "Hilda", "Hilary", "Holly", "Howard", "Idris",
    "Ilona", "Imani", "India", "Indira", "Inez", "Inga", "Ismail", "Jacinta", "Jacqueline",
    "Jamal", "Jared", "Jason", "Jean", "Jeffrey", "Jemima", "Jeremy", "Jessica", "Joanna",
    "Jocelyn", "Jonah", "Jonathan", "Josephine", "Joshua", "Judith", "Justin", "Kaden", "Kahlil",
    "Kaitlyn", "Kamila", "Karina", "Katerina", "Keira", "Kendall", "Kenneth", "Khaled", "Khadija",
    "Kiara", "Kirsten", "Kristen", "Lachlan", "Lana", "Laurence", "Lawrence", "Layton", "Leah",
    "Leandro", "Leonard", "Leonie", "Leslie", "Liana", "Lilian", "Lionel", "Lisa", "Lola",
    "Lorelei", "Lorna", "Louis", "Lucinda", "Luis", "Mabel", "Macy", "Malcolm", "Manon", "Marcel",
    "Marcia", "Marcus", "Margaret", "Marisol", "Marjorie", "Martha", "Mary", "Maureen", "Maxwell",
    "Melina", "Melissa", "Meredith", "Micah", "Mildred", "Miranda", "Miriam", "Mitchell", "Moira",
    "Monty", "Mustafa", "Myles", "Nadine", "Naomi", "Natalia", "Nell", "Nessa", "Nicholas", "Nina",
    "Noel", "Nolan", "Octavia", "Odette", "Opal", "Orla", "Orlando", "Orson", "Paige", "Patricia",
    "Penelope", "Percy", "Phoebe", "Pranav", "Quentin", "Rachid", "Ramona", "Raphael", "Ray",
    "Raymond", "Regina", "Reuben", "Rex", "Rhiannon", "Rhoda", "Rina", "Rita", "Rocco", "Roland",
    "Ronan", "Rosalie", "Rosalind", "Rose", "Rosemary", "Rowan", "Ruben", "Russell", "Sabrina",
    "Sadie", "Safiya", "Sally", "Samson", "Sandra", "Savannah", "Selina", "Serena", "Shreya",
    "Sidney", "Simone", "Solomon", "Sonia", "Spencer", "Stanley", "Stephanie", "Summer", "Sylvia",
    "Tabitha", "Tamara", "Tanya", "Tatiana", "Teagan", "Terence", "Thea", "Theodore", "Theresa",
    "Tristan", "Troy", "Uday", "Ugo", "Ulrich", "Usha", "Veda", "Vera", "Veronica", "Vincent",
    "Violet", "Virginia", "Wallace", "Warren", "Wesley", "Wilfred", "Winston", "Yasir", "Yvette",
    "Zahra", "Zain", "Zola",
)
_FAMILY_OWNERS = (
    "Mum's", "Mom's", "Dad's", "Grandma's", "Grandpa's", "Grandad's", "Gran's", "Nana's", "Papa's",
    "Auntie's", "Uncle's", "Baby's", "Kids'", "Children's", "Nonna's",
)
_OWNERS = (
    "O'Malley's", "McKay's",
    *_FAMILY_OWNERS,
    *(f"{name}'" if name.endswith("s") else f"{name}'s" for name in _FIRST_NAMES),
)
_OUTDOOR_AREAS = (
    "Garden", "Backyard", "Front Yard", "Driveway", "Walkway", "Porch", "Patio", "Deck", "Balcony",
    "Courtyard", "Carport", "Gazebo", "Pool Deck", "Side Yard", "Rooftop", "Pergola", "Lanai",
    "Fire Pit", "Back Porch", "Vegetable Patch",
)
_DEVICE_AREAS = {
    "media_players": (
        "Living Room", "Family Room", "Den", "Lounge", "Master Bedroom", "Guest Bedroom",
        "Kids Room", "Media Room", "Home Theater", "Basement", "Games Room", "Bonus Room",
        "Teen Room", "Guest House", "Pool House",
    ),
    "climate": (
        "Hallway", "Living Room", "Master Bedroom", "Guest Bedroom", "Office", "Basement",
        "Front Bedroom", "Back Bedroom", "Granny Flat", "Guest House",
    ),
    "vacuums": (
        "Living Room", "Hallway", "Utility Room", "Laundry Room", "Kitchen", "Games Room",
        "Playroom", "Bonus Room",
    ),
    "locks": (
        "Entryway", "Foyer", "Garage", "Porch", "Mudroom", "Back Porch",
    ),
    "lawn_mowers": (
        "Garden", "Backyard", "Front Yard", "Shed", "Garage", "Side Yard",
    ),
    "todo_lists": (
        "Kitchen", "Office", "Utility Room", "Pantry", "Scullery",
    ),
}


# Holdout-only vocabulary. Never used by default generation; holdout homes draw
# exclusively from these, which keeps them disjoint from every training pool.
_HOLDOUT_INDOOR_AREAS = (
    "Conservatory", "Snug", "Boot Room", "Loft", "Wine Cellar",
    "Gallery", "Ballroom", "Cinema Room", "Vestibule", "Potting Shed",
)
_HOLDOUT_OUTDOOR_AREAS = ("Orchard", "Terrace", "Allotment", "Greenhouse")
_HOLDOUT_AREAS = (*_HOLDOUT_INDOOR_AREAS, *_HOLDOUT_OUTDOOR_AREAS)
_HOLDOUT_DEVICE_AREAS = {
    "media_players": ("Snug", "Gallery", "Ballroom", "Cinema Room", "Loft"),
    "climate": ("Conservatory", "Loft", "Snug"),
    "vacuums": ("Boot Room", "Conservatory", "Wine Cellar"),
    "locks": ("Vestibule", "Boot Room"),
    "lawn_mowers": ("Orchard", "Allotment", "Greenhouse", "Terrace"),
    "todo_lists": ("Conservatory", "Boot Room"),
}
_HOLDOUT_ROLES = {
    "lights": ("Sconce", "Strip Light", "Uplight", "Pendant Sconce"),
    "fans": ("Fan Heater", "Pedestal Fan", "Bladeless Fan"),
    "switches": ("Smart Plug", "Extension Socket", "Mains Relay"),
    "covers": ("Roman Blind", "Shutter", "Venetian Blind"),
    "locks": ("Keypad Lock", "Smart Deadbolt", "Gate Latch"),
    "media_players": ("Projector", "Soundbar Pair", "Stereo Amplifier", "Bluetooth Speaker"),
    "climate": ("Radiator Valve", "Underfloor Heating", "Heat Recovery Unit"),
    "vacuums": ("Mop Robot", "Wet Vacuum"),
    "scripts": ("Sunrise Routine", "Guest Departure", "Shutdown Sequence", "Cellar Check"),
    "scenes": ("Cinema Mode", "Candlelight", "Brunch Setting", "Guest Arrival"),
    "lawn_mowers": ("Ride-on Mower", "Robot Scythe", "Petrol Mower"),
    "todo_lists": ("Garden Centre", "Pharmacy Run", "Dry Cleaning"),
    "buttons": ("Panic Button", "Scene Dial"),
}
_HOLDOUT_BRANDS = {
    "lights": ("Philips", "Yeelight"),
    "fans": ("Meaco",),
    "switches": ("Meross", "Wemo"),
    "covers": ("Eve",),
    "locks": ("Nuki",),
    "media_players": ("Epson", "Bose", "Denon"),
    "climate": ("Daikin", "Mitsubishi"),
    "vacuums": ("Dreame", "Shark"),
}
_HOLDOUT_OWNERS = ("Tariq's", "Ngozi's", "Saoirse's", "Ravi's", "Elif's")


def device_areas(capability: str, areas: list[str], *, holdout: bool = False) -> list[str]:
    allowed = (_HOLDOUT_DEVICE_AREAS if holdout else _DEVICE_AREAS).get(capability)
    if allowed:
        return [area for area in areas if area in allowed]
    outdoor = _HOLDOUT_OUTDOOR_AREAS if holdout else _OUTDOOR_AREAS
    if capability in {"covers", "fans"}:
        return [area for area in areas if area not in outdoor]
    return areas


_LOUNGE_SCENES = (
    "Movie Night", "Reading Lights", "Relax", "Evening Lights", "Game Night", "Cozy Lights",
    "TV Time", "Bright Lights", "Dimmed Lights", "Night Lights", "Late Night", "Warm Glow",
)
_DINING_SCENES = (
    "Dinner Lights", "Bright Lights", "Evening Lights", "Night Lights", "Dinner Time",
    "Lunch Lights", "Snack Time", "Baking", "After Dinner", "Late Night", "Warm Glow",
    "Dimmed Lights",
)
_ROOM_SCENES = (
    "Dimmed Lights", "Bright Lights", "Evening Lights", "Night Lights", "Warm Glow", "Cozy Lights",
    "Late Night", "Focus Lights", "Soft Lights", "Wind Down", "All Off", "Daylight",
)
_EXTRACTOR_FANS = (
    "Extractor Fan", "Ventilation Fan", "Exhaust Fan", "Range Hood Fan", "Vent Fan", "Inline Fan",
    "Bath Fan", "Hood Fan", "Ceiling Vent",
)
_KITCHEN_SWITCHES = (
    "Coffee Maker Plug", "Kettle Plug", "Counter Outlet", "Toaster Plug", "Air Fryer Plug",
    "Blender Plug", "Microwave Plug", "Dishwasher Plug", "Fridge Plug", "Stand Mixer Plug",
    "Bread Maker Plug", "Juicer Plug",
)


def _roles(capability: str, area: str) -> tuple[str, ...]:
    if capability == "scenes":
        if area in {"Living Room", "Family Room", "Den", "Lounge", "Media Room", "Home Theater",
                    "Games Room", "Bonus Room"}:
            return _LOUNGE_SCENES
        if area in {"Kitchen", "Dining Room", "Dining Area"}:
            return _DINING_SCENES
        return _ROOM_SCENES
    if capability == "lights":
        if area in _OUTDOOR_AREAS:
            return _ROOM_LIGHTS["Outside"]
        if "Bedroom" in area or area in {"Kids Room", "Nursery", "Teen Room"}:
            return _ROOM_LIGHTS["Bedroom"]
        if area in {"Bathroom", "Ensuite", "Powder Room", "Washroom"}:
            return _ROOM_LIGHTS["Bathroom"]
        return _ROOM_LIGHTS.get(area, _ROLES[capability])
    if capability == "fans" and area in {"Kitchen", "Bathroom", "Ensuite", "Laundry Room",
                                         "Scullery", "Powder Room", "Utility Room"}:
        return _EXTRACTOR_FANS
    if capability == "switches" and area == "Kitchen":
        return _KITCHEN_SWITCHES
    return _ROLES[capability]


_ENTITY_TEMPLATES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "lights": ("Light", ("on", "off"), ("on", "off", "brightness", "color", "color_temp")),
    "fans": ("Fan", ("on", "off"), ("on", "off", "percentage")),
    "switches": ("Outlet", ("on", "off"), ("on", "off")),
    "covers": ("Blinds", ("open", "closed"), ("open", "close")),
    "locks": ("Door Lock", ("locked", "unlocked"), ("lock", "unlock")),
    "media_players": ("TV", ("off", "idle", "playing"),
                      ("on", "off", "play", "pause", "volume", "volume_step", "mute",
                       "next", "previous", "search")),
    "climate": ("Thermostat", ("heat", "cool", "off"), ("heat", "cool", "off")),
    "vacuums": ("Vacuum", ("docked", "cleaning"), ("start", "stop")),
    "scenes": ("Scene", ("off", "on"), ("activate",)),
    "scripts": ("Script", ("off", "on"), ("run",)),
    "lawn_mowers": ("Lawn Mower", ("docked", "mowing"), ("start", "stop")),
    "todo_lists": ("Shopping List", ("idle",), ("add",)),
    "buttons": ("Button", ("idle",), ("press",)),
}

_SPECIAL_NAMES: dict[str, tuple[str, ...]] = {
    "lights": ("Kids' Room Light", "O'Malley's Porch Light"),
    "fans": ("Joe's Workshop Fan", "Children's Bedroom Fan"),
    "switches": ("McKay's Office Outlet", "Joe's Desk Outlet"),
    "covers": ("Children's Bedroom Blinds", "O'Malley's Study Blinds"),
    "locks": ("McKay's Front Door Lock", "Children's Door Lock"),
    "media_players": ("Living Room TV", "Bedroom TV"),
}

_KIND_MAP: dict[str, str] = {
    "lights": "light",
    "fans": "fan",
    "switches": "switch",
    "covers": "cover",
    "locks": "lock",
    "media_players": "media_player",
    "climate": "climate",
    "vacuums": "vacuum",
    "scenes": "scene",
    "scripts": "script",
    "lawn_mowers": "lawn_mower",
    "todo_lists": "todo",
    "buttons": "button",
    "timers": "timer",
}


_NICKNAMES = {
    "ceiling lights": (
        "big lights", "overhead lights", "main lights",
    ),
    "ceiling light": (
        "big light", "overhead light", "main light",
    ),
    "counter lights": (
        "worktop lights", "cooking lights",
    ),
    "bedside lamp": (
        "night lamp", "nightstand lamp",
    ),
    "floor lamp": (
        "corner lamp", "standing lamp",
    ),
    "reading lamp": (
        "book lamp", "chair lamp",
    ),
    "desk lamp": (
        "work lamp", "study lamp",
    ),
    "table lamp": (
        "side lamp",
    ),
    "tv": (
        "telly", "big screen",
    ),
    "television": (
        "telly", "big screen",
    ),
    "speaker": (
        "music speaker",
    ),
    "ceiling fan": (
        "overhead fan",
    ),
    "desk fan": (
        "cooling fan",
    ),
    "floor fan": (
        "cooling fan",
    ),
    "tower fan": (
        "cooling fan",
    ),
    "kettle plug": (
        "kettle",
    ),
    "coffee maker plug": (
        "coffee machine",
    ),
    "charger plug": (
        "phone charger",
    ),
    "downlights": (
        "pot lights", "can lights",
    ),
    "pendant lights": (
        "hanging lights",
    ),
    "chandelier": (
        "hanging light",
    ),
    "string lights": (
        "twinkle lights",
    ),
    "floodlight": (
        "flood lamp",
    ),
    "box fan": (
        "cooling fan",
    ),
    "table fan": (
        "cooling fan",
    ),
    "air fryer plug": (
        "air fryer",
    ),
    "microwave plug": (
        "microwave",
    ),
    "dishwasher plug": (
        "dishwasher",
    ),
    "heater plug": (
        "space heater",
    ),
    "router plug": (
        "wifi router",
    ),
    "aquarium plug": (
        "fish tank",
    ),
    "soundbar": (
        "tv speaker",
    ),
    "robot vacuum": (
        "robot",
    ),
    "thermostat": (
        "heating control",
    ),
    "stand mixer plug": (
        "mixer",
    ),
    "bread maker plug": (
        "bread machine",
    ),
    "juicer plug": (
        "juicer",
    ),
    "bookcase lights": (
        "book lights",
    ),
    "headboard light": (
        "pillow light",
    ),
}
_NICKNAME_KEYS = sorted(_NICKNAMES, key=len, reverse=True)


def _nickname_alias(
    name: str, owners: tuple[str, ...], nickname_rng: random.Random, used: set[str]
) -> str | None:
    lowered = name.casefold()
    for key in _NICKNAME_KEYS:
        if not lowered.endswith(key):
            continue
        if nickname_rng.random() >= 0.5:
            return None
        if any(lowered.startswith(owner.casefold()) for owner in owners):
            candidates = (f"my {key}",)
        else:
            candidates = _NICKNAMES[key]
        for candidate in candidates:
            if candidate.casefold() not in used:
                return candidate
        return None
    return None


def _slug(name: str) -> str:
    return "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")


def _device_class_for(cap: CapabilitySpec) -> str | None:
    mapping = {
        "covers": "blind",
        "locks": "door",
        "switches": "outlet",
        "media_players": "tv",
    }
    return cap.device_class or mapping.get(cap.name)


_MEDIA_PROFILES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        ("satellite",),
        "speaker",
        # No absolute "volume": a satellite is only adjusted relative to its current level.
        ("on", "off", "play", "pause", "volume_step", "mute"),
    ),
    (
        ("echo", "smart speaker", "google nest", "homepod", "nest mini"),
        "speaker",
        ("play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
    (
        ("speaker", "soundbar", "sonos", "stereo", "receiver", "subwoofer", "radio"),
        "speaker",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous"),
    ),
    (
        ("streaming tv", "roku", "apple tv", "chromecast", "shield", "fire tv", "streaming stick"),
        "tv",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
)
_MEDIA_DEFAULT = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")


def media_player_profile(name: str) -> tuple[str, tuple[str, ...]]:
    lowered = name.casefold()
    for keywords, device_class, features in _MEDIA_PROFILES:
        if any(keyword in lowered for keyword in keywords):
            return device_class, features
    return "tv", _MEDIA_DEFAULT


def make_entity(
    *,
    name: str,
    capability: str,
    area: str,
    floor: str,
    rng: random.Random,
    aliases: list[str] | None = None,
    features: tuple[str, ...] | None = None,
    state: str | None = None,
    device_class: str | None = None,
) -> dict[str, Any]:
    cap = CAPABILITIES[capability]
    kind = _KIND_MAP[capability]
    noun, states, default_features = _ENTITY_TEMPLATES[capability]
    domain = cap.domain
    if capability == "media_players" and (device_class is None or features is None):
        profiled_class, profiled_features = media_player_profile(name)
        device_class = device_class or profiled_class
        features = features or profiled_features
    device_class = device_class or _device_class_for(cap)
    return {
        "entity_id": f"{domain}.{_slug(name)}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "capability": capability,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": state or rng.choice(states),
        "capabilities": list(features or default_features),
    }


def _eval_cases_module() -> Any:
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    try:
        from evals import cases
    except Exception:
        return None
    return cases


@lru_cache(maxsize=1)
def _eval_entity_names() -> frozenset[str]:
    # Every eval case's household, every tag, names and aliases: a synthetic name equal to any of
    # these makes the command that names it fail the generation-time eval-overlap check.
    cases = _eval_cases_module()
    if cases is None:
        return frozenset()
    names: set[str] = set()
    for household in {case.household for case in cases.load_all_cases()}:
        for entity in cases.load_home(household).get("entities") or []:
            names.add(entity["name"])
            names.update(entity.get("aliases") or [])
    return frozenset(name.casefold() for name in names)


_HOLDOUT = False


@contextmanager
def holdout_pools() -> Iterator[None]:
    global _HOLDOUT
    previous, _HOLDOUT = _HOLDOUT, True
    try:
        yield
    finally:
        _HOLDOUT = previous


def _holdout_mode(holdout: bool | None) -> bool:
    return _HOLDOUT if holdout is None else holdout


def holdout_slice() -> dict[str, frozenset[str]]:
    return {
        "areas": frozenset(_HOLDOUT_AREAS),
        "nouns": frozenset(noun for nouns in _HOLDOUT_ROLES.values() for noun in nouns),
        "owners": frozenset(_HOLDOUT_OWNERS),
        "brands": frozenset(brand for brands in _HOLDOUT_BRANDS.values() for brand in brands),
    }


_CONTEXT_PERSON_NAMES = (
    "Alice Morgan", "Ben Carter", "Chloe Nguyen", "David Osei", "Emma Lindqvist", "Farid Haddad",
    "Grace Kim", "Hugo Martins", "Ines Duarte", "Jonas Weber", "Adam Kowalski", "Abigail Reyes",
    "Aidan Murphy", "Akosua Mensah", "Alan Brooks", "Alicia Romero", "Amelia Clarke",
    "Andre Dubois", "Andrea Moretti", "Angela Ruiz", "Anika Shah", "Anton Petrov", "Ariana Vega",
    "Arnav Mehta", "Asha Pillai", "Aurora Lindgren", "Ayumi Tanaka", "Basil Hughes",
    "Benedikt Huber", "Bianca Rossi", "Brendan Walsh", "Bruno Costa", "Callum Fraser",
    "Carla Mendes", "Caroline Price", "Cecilia Lund", "Chiamaka Obi", "Christina Novak",
    "Cian Byrne", "Claudia Ferrari", "Colin Grant", "Connor Doyle", "Daniela Silva",
    "Daniel Fischer", "Darius Ahmadi", "Delia Popescu", "Dennis Wagner", "Dina Cohen",
    "Dominic Reid", "Edgar Alves", "Edith Barnes", "Eduardo Ramos", "Elena Ivanova", "Elias Berg",
    "Eliza Ward", "Ella Thompson", "Emil Nowak", "Eric Larsen", "Esther Adeyemi", "Ethan Cooper",
    "Eva Horvat", "Fabian Keller", "Fiona Campbell", "Francis Okoro", "Frida Holm",
    "Gabriel Santos", "Gemma Lewis", "George Evans", "Gloria Mwangi", "Gustav Nyberg",
    "Hamza Malik", "Hannah Becker", "Harold Simmons", "Helen Turner", "Ibrahim Diallo",
    "Irene Papadopoulos", "Isaac Levi", "Isabel Navarro", "Ivan Horak", "Jack Bennett",
    "Jacob Stein", "Jade Wong", "James Patel", "Jana Novotna", "Jasmine Ali", "Jay Sharma",
    "Jennifer Scott", "Joel Lindberg", "Jordan Hayes", "Jose Herrera", "Joseph Mbeki",
    "Julia Hoffmann", "Julian Wolf", "June Park", "Karen Fox", "Karl Jensen", "Kate Sullivan",
    "Kayode Bello", "Keith Morris", "Kevin Tran", "Laura Bianchi", "Leo Schmidt", "Liam O'Brien",
    "Lily Chen", "Lina Saleh", "Logan Pierce", "Louise Garnier", "Lucas Moreau", "Lydia Hale",
    "Madeleine Roux", "Magnus Eriksen", "Maria Gonzalez", "Mark Davies", "Martin Kral",
    "Maya Rosen", "Megan Hart", "Michael Brown", "Mila Kovac", "Mohammed Rahman", "Monica Lopez",
    "Nathan Wood", "Neil Kapoor", "Nicole Baker", "Noah Friedman", "Olivia Hunt", "Owen Jenkins",
    "Patrick Kelly", "Paula Ortiz", "Paul Richter", "Peter Holmberg", "Rachel Stone",
    "Rebecca Marsh", "Ricardo Pereira", "Robert Klein", "Ruth Abrams", "Ryan Foster",
    "Sara Lindahl", "Sean Gallagher", "Sebastian Vogel", "Simon Ayers", "Sophie Laurent",
    "Stefan Popov", "Stella Marino", "Stephen Quinn", "Susan Webb", "Teresa Castillo",
    "Thomas Muller", "Tim Hartmann", "Tobias Lange", "Uma Krishnan", "Ursula Brandt",
    "Valerie Dumont", "Vanessa Cruz", "Victor Lindholm", "Vivian Cho", "Walter Graham",
    "Wendy Lam", "William Ford", "Xavier Blanc", "Yara Nassar", "Yvonne Bauer", "Zachary Cole",
    "Zara Hussain", "Zoe Parsons", "Joe Adams", "Priya Adkins", "Wei Aguilar", "Amara Ahmed",
    "Sofia Akhtar", "Yusuf Albertson", "Ingrid Alford", "Mateo Allison", "Rosa Andersen",
    "Dmitri Anderson", "Fatima Andrade", "Kenji Andrews", "Niamh Anthony", "Tomas Archer",
    "Leila Arellano", "Bram Arias", "Oona Armstrong", "Hallie Arnold", "Noor Arroyo",
    "Ellis Ashley", "Hana Atkinson", "Ivy Atkins", "Dev Avila", "Leona Ayala", "Oscar Baird",
    "Wren Baldwin", "Kwame Ballard", "Ama Banks", "Kofi Barrett", "Abena Barron", "Chidi Barry",
    "Adaeze Bates", "Emeka Baxter", "Folake Bean", "Tunde Beard", "Yetunde Beck", "Sipho Bell",
    "Thandi Beltran", "Zanele Benjamin", "Lindiwe Benson", "Bongani Bentley", "Amani Benton",
    "Jabari Berger", "Zuri Bernard", "Wanjiru Berry", "Kamau Bishop", "Nia Blair",
    "Oluwaseun Blake", "Ayodele Blankenship", "Kehinde Bolton", "Mandla Bond", "Nomvula Booker",
    "Dayo Boone", "Arjun Booth", "Ananya Bowen", "Rohan Boyd", "Kavya Bradford", "Vikram Brady",
    "Meera Brennan", "Aarav Brewer", "Ishaan Bridges", "Diya Briggs", "Sanjay Brock",
    "Lakshmi Browning", "Aditi Bruce", "Nikhil Bryant", "Pooja Buchanan", "Rahul Buckley",
    "Sunita Bullock", "Anil Burgess", "Harpreet Burke", "Gurpreet Burns", "Imran Burton",
    "Ayesha Bush", "Zainab Butler", "Bilal Cabrera", "Farhan Cain", "Nadia Caldwell",
    "Saanvi Cameron", "Kiran Campos", "Mei Cannon", "Jun Cantrell", "Hiroshi Cardenas",
    "Yuki Carey", "Sakura Carlson", "Haruto Carr", "Aiko Carroll", "Ren Carson",
    "Jiwoo Cartwright", "Hyun Castro", "Jisoo Cervantes", "Minho Chandler", "Xiu Chang",
    "Ming Chapman", "Lan Chavez", "Bao Christensen", "Linh Christian", "Minh Church",
    "Thao Cisneros", "Duc Clayton", "Hoa Clements", "Akira Cleveland", "Kaito Cobb",
    "Haruka Cochran", "Daisuke Coker", "Emi Coleman", "Takumi Collier", "Jia Collins",
    "Sora Combs", "Riku Conley", "Omar Conrad", "Layla Contreras", "Hassan Conway", "Rania Cook",
    "Karim Copeland", "Yasmin Cortez", "Samir Cote", "Dalia Cotton", "Mona Craig", "Ziad Crawford",
    "Hiba Crosby", "Nabil Cummings", "Salma Cunningham", "Amir Curry", "Soraya Curtis",
    "Dariush Dalton", "Shirin Daniels", "Reza Daugherty", "Parisa Davenport", "Cyrus Davidson",
    "Mehmet Dawson", "Ayse Decker", "Emre Delgado", "Zeynep Dempsey", "Deniz Dennis",
    "Burak Denton", "Lars Dickerson", "Astrid Dickson", "Freya Dodson", "Sven Dominguez",
    "Bjorn Donaldson", "Sigrid Donovan", "Henrik Dorsey", "Elin Douglas", "Mikkel Drake",
    "Signe Dudley", "Aino Duffy", "Eero Duke", "Pieter Duncan", "Anouk Dunlap", "Joost Dunn",
    "Lotte Duran", "Matthias Durham", "Greta Dyer", "Lukas Eaton", "Anja Edwards", "Klaus Elliott",
    "Heidi Ellison", "Pierre Emerson", "Camille English", "Margaux Erickson", "Lucien Escobar",
    "Elodie Espinoza", "Giulia Estes", "Luca Estrada", "Marco Everett", "Chiara Ewing",
    "Alessandro Farley", "Francesca Farmer", "Lorenzo Farrell", "Carmen Faulkner",
    "Javier Ferguson", "Lucia Fernandez", "Pablo Fields", "Joao Figueroa", "Beatriz Finley",
    "Tiago Fitzpatrick", "Agnieszka Fleming", "Piotr Fletcher", "Kasia Flores", "Marek Floyd",
    "Zofia Flynn", "Ondrej Foley", "Petra Forbes", "Milan Fowler", "Ivana Francis",
    "Nikola Franklin", "Dragan Frazier", "Anastasia Freeman", "Sergei French", "Olga Frost",
    "Natasha Fuentes", "Yuri Fuller", "Katya Gaines", "Bogdan Gallegos", "Oksana Galloway",
    "Taras Gamble", "Eleni Garcia", "Nikos Gardner", "Yiannis Garza", "Cormac Gates",
    "Siobhan Gay", "Aoife Gentry", "Ciaran Gibbs", "Declan Gibson", "Eilidh Giles", "Morag Gill",
    "Hamish Gillespie", "Rhys Gilmore", "Cerys Glass", "Dylan Glenn", "Bethan Glover",
    "Gethin Golden", "Diego Good", "Valentina Goodman", "Camila Goodwin", "Santiago Gordon",
    "Ximena Gould", "Joaquin Grady", "Paloma Graves", "Rafael Gray", "Guadalupe Green",
    "Alejandro Greene", "Renata Griffin", "Thiago Griffith", "Luana Grimes", "Caio Gross",
    "Oliver Guerra", "Harriet Guerrero", "Felix Gutierrez", "Imogen Guzman", "Theo Haas",
    "Poppy Hahn", "Arthur Haley", "Florence Hamilton", "Jasper Hammond", "Matilda Hampton",
    "Rupert Hancock", "Beatrix Hanna", "Hazel Hansen", "Miles Hanson", "Nora Harding",
    "Silas Hardy", "Clara Harmon", "Ezra Harper", "Iris Harrington", "Jude Harris",
    "Maeve Harrison", "Rory Harvey", "Tessa Hawkins", "Wyatt Haynes", "Hattie Heath",
    "Otis Hebert", "Ada Henderson", "Bea Hendricks", "Elsie Hendrix", "Archie Henry",
    "Aroha Hensley", "Tane Henson", "Mere Herman", "Wiremu Hernandez", "Keanu Herring",
    "Leilani Hess", "Malia Hester", "Kai Hicks", "Abdul Higgins", "Abdullah Hill", "Abel Hines",
    "Abraham Hinton", "Adeline Hodge", "Adrian Hodges", "Adrienne Hogan", "Agatha Holden",
    "Ahmed Holder", "Aisha Holland", "Albert Holloway", "Alberto Holt", "Alessia Hood",
    "Alfred Hoover", "Alina Hopkins", "Alistair Horne", "Althea Horton", "Amelie House",
    "Amos Houston", "Amrita Howard", "Anais Howell", "Anders Hubbard", "Angel Hudson",
    "Ansel Huff", "Anushka Huffman", "Antonio Hull", "April Humphrey", "Ari Hunter",
    "Ariel Hurley", "Arlo Hurst", "Arman Hutchinson", "Armando Ingram", "Asher Irwin",
    "Asma Jackson", "Athena Jacobs", "Augustin Jarvis", "Austin Jefferson", "Ava Jennings",
    "Axel Jimenez", "Aylin Johns", "Aziza Johnson", "Barbara Johnston", "Beatrice Jones",
    "Benedict Jordan", "Bernadette Joyce", "Bernard Juarez", "Beth Kane", "Betty Kaufman",
    "Beverly Keating", "Blake Kemp", "Blythe Kennedy", "Bonnie Kent", "Boris Kerr", "Brandon Key",
    "Brenda Kidd", "Brian King", "Bridget Kirby", "Brittany Kirk", "Brody Kirkland",
    "Caleb Kirkpatrick", "Calvin Knapp", "Cassandra Knight", "Catherine Knox", "Cedric Koch",
    "Celeste Kramer", "Celia Lamb", "Charles Lambert", "Charlotte Lancaster", "Chester Lane",
    "Christian Lang", "Christopher Langley", "Clarence Lara", "Clement Larson",
    "Clementine Lawson", "Cora Leach", "Corinne Leblanc", "Cosima Leon", "Curtis Lester",
    "Cynthia Leung", "Daisy Levine", "Dakota Li", "Damian Lim", "Daphne Little",
    "Deborah Livingston", "Desmond Lloyd", "Diana Long", "Dolores Love", "Donovan Lovell",
    "Dorothy Lowery", "Douglas Lozano", "Duncan Luna", "Eamon Lynch", "Eden Lynn", "Edmund Lyons",
    "Edward Macdonald", "Edwin Mack", "Eileen Madden", "Eleanor Maddox", "Elisa Maldonado",
    "Elizabeth Malone", "Eloise Mann", "Emanuel Manning", "Emilia Mansfield", "Emily Marks",
    "Enid Marshall", "Enzo Martinez", "Ernest Massey", "Esme Mathews", "Estelle Mathis",
    "Eugene Matthews", "Evelyn Maxwell", "Everett May", "Faisal Mayer", "Faye Maynard",
    "Fern Mays", "Finlay McBride", "Finn McCarthy", "Flora McClain", "Frances McCormick",
    "Frederick McCoy", "Gabriela McCullough", "Gareth McDaniel", "Gavin McDonald",
    "Georgina McDowell", "Gerald McGee", "Geraldine McGuire", "Gideon McIntosh",
    "Gilbert McKenzie", "Giovanna McKinney", "Giselle McLaughlin", "Gordon McLean",
    "Gregory McMahon", "Gwen Medina", "Hadley Melton", "Hakeem Mendoza", "Hanan Mercer",
    "Harlow Merritt", "Harry Meyer", "Harvey Meyers", "Hector Middleton", "Helena Miles",
    "Hilda Miller", "Hilary Mills", "Holly Miranda", "Howard Mitchell", "Idris Molina",
    "Ilona Monroe", "Imani Montgomery", "India Moody", "Indira Moore", "Inez Moran", "Inga Moreno",
    "Ismail Morton", "Jacinta Moses", "Jacqueline Moss", "Jamal Moyer", "Jared Mullen",
    "Jason Mullins", "Jean Munoz", "Jeffrey Murray", "Jemima Nash", "Jeremy Neal",
    "Jessica Nelson", "Joanna Newman", "Jocelyn Newton", "Jonah Nichols", "Jonathan Nicholson",
    "Josephine Nielsen", "Joshua Noble", "Judith Norman", "Justin Norris", "Kaden Norton",
    "Kahlil Nunez", "Kaitlyn Obrien", "Kamila Ochoa", "Karina Oliver", "Katerina Olsen",
    "Keira Olson", "Kendall Oneill", "Kenneth Osborne", "Khaled Owens", "Khadija Pace",
    "Kiara Pacheco", "Kirsten Padilla", "Kristen Page", "Lachlan Palmer", "Lana Parker",
    "Laurence Parks", "Lawrence Parrish", "Layton Parsons", "Leah Pate", "Leandro Patrick",
    "Leonard Patterson", "Leonie Patton", "Leslie Payne", "Liana Pearson", "Lilian Peck",
    "Lionel Pena", "Lisa Perez", "Lola Perkins", "Lorelei Perry", "Lorna Peters", "Louis Petersen",
    "Lucinda Peterson", "Luis Phelps", "Mabel Phillips", "Macy Pickett", "Malcolm Pittman",
    "Manon Pitts", "Marcel Pollard", "Marcia Poole", "Marcus Pope", "Margaret Porter",
    "Marisol Potts", "Marjorie Powell", "Martha Pratt", "Mary Preston", "Maureen Pruitt",
    "Maxwell Puckett", "Melina Pugh", "Melissa Raines", "Meredith Ramsey", "Micah Randolph",
    "Mildred Rasmussen", "Miranda Ratliff", "Miriam Ray", "Mitchell Raymond", "Moira Reese",
    "Monty Reeves", "Mustafa Reid", "Myles Reilly", "Nadine Reynolds", "Naomi Rhoades",
    "Natalia Rhodes", "Nell Rice", "Nessa Rich", "Nicholas Richards", "Nina Richardson",
    "Noel Riddle", "Nolan Riggs", "Octavia Riley", "Odette Rios", "Opal Rivas", "Orla Rivers",
    "Orlando Roach", "Orson Robbins", "Paige Roberson", "Patricia Roberts", "Penelope Robertson",
    "Percy Robinson", "Phoebe Rocha", "Pranav Rodgers", "Quentin Rodriguez", "Rachid Rogers",
    "Ramona Rollins", "Raphael Roman", "Ray Rosales", "Raymond Rosario", "Regina Rose",
    "Reuben Ross", "Rex Roth", "Rhiannon Rowe", "Rhoda Rowland", "Rina Roy", "Rita Rubio",
    "Rocco Rucker", "Roland Rudolph", "Ronan Ruiz", "Rosalie Rush", "Rosalind Russell",
    "Rose Russo", "Rosemary Rutledge", "Rowan Salazar", "Ruben Salinas", "Russell Sanchez",
    "Sabrina Sanders", "Sadie Sanford", "Safiya Santana", "Sally Santiago", "Samson Saunders",
    "Sandra Savage", "Savannah Sawyer", "Selina Schaefer", "Serena Schneider", "Shreya Schroeder",
    "Sidney Schultz", "Simone Schwartz", "Solomon Sellers", "Sonia Serrano", "Spencer Sexton",
    "Stanley Shaffer", "Stephanie Shannon", "Summer Sharp", "Sylvia Shaw", "Tabitha Shelton",
    "Tamara Shepard", "Tanya Shepherd", "Tatiana Sherman", "Teagan Shields", "Terence Short",
    "Thea Silva", "Theodore Simmons", "Theresa Simpson", "Tristan Sims", "Troy Singleton",
    "Uday Skinner", "Ugo Slater", "Ulrich Sloan", "Usha Small", "Veda Smith", "Vera Snyder",
    "Veronica Solis", "Vincent Solomon", "Violet Soto", "Virginia Sparks", "Wallace Spears",
    "Warren Spencer", "Wesley Stafford", "Wilfred Stanley", "Winston Stanton", "Yasir Stark",
    "Yvette Steele", "Zahra Stephens", "Zain Stevenson", "Zola Stevens",
)


def _context_entities_for(
    entities: list[dict[str, Any]], rooms: list[str], index: int, size: int,
    rng_state: tuple[int, tuple[int, ...], None | Any]
) -> list[dict[str, Any]]:
    rng = random.Random(f"context:{index}:{size}:{rng_state}")
    taken: set[str] = set()
    for entity in entities:
        taken.add(entity["name"].casefold())
        for alias in entity.get("aliases", []):
            taken.add(alias.casefold())
    reserved = _eval_entity_names()
    items: list[dict[str, Any]] = []

    def add(name: str, domain: str, area: str | None = None) -> None:
        if name.casefold() in taken or name.casefold() in reserved:
            return
        taken.add(name.casefold())
        item: dict[str, Any] = {"name": name, "domain": domain}
        if area:
            item["area"] = area
        items.append(item)

    for entity in entities:
        if entity["capability"] == "media_players" and rng.random() < 0.8:
            add(f"{entity['name']} Active app", "sensor", entity.get("area"))
            add(f"{entity['name']} Application", "select", entity.get("area"))
    for entity in entities:
        if entity["capability"] in {"lights", "switches"} and rng.random() < 0.15:
            add(f"{entity['name']} {rng.choice(('Power', 'Energy'))}", "sensor", entity.get("area"))
    for area in rooms:
        if rng.random() < 0.25:
            add(f"{area} Temperature", "sensor", area)
            add(f"{area} Humidity", "sensor", area)
            add(f"{area} Motion", "binary_sensor", area)
    for name in rng.sample(_CONTEXT_PERSON_NAMES, rng.randint(1, 3)):
        add(name, "person")
    if rng.random() < 0.9:
        add("Sun", "sun")
    if rng.random() < 0.85:
        add("Date", "sensor")
    return items


def _random_entity_name(
    capability: str,
    area: str,
    slot: int,
    index: int,
    rng: random.Random,
    taken: set[str] | None = None,
    owners: tuple[str, ...] | None = None,
    bare_name_rate: float = 0.0,
    *,
    holdout: bool | None = None,
) -> str:
    holdout = _holdout_mode(holdout)
    taken = taken if taken is not None else set()
    if holdout:
        owners = owners if owners is not None else _HOLDOUT_OWNERS
        roles = _HOLDOUT_ROLES[capability]
        brands = _HOLDOUT_BRANDS.get(capability, ())
    else:
        owners = owners if owners is not None else _OWNERS
        roles = _roles(capability, area)
        brands = _BRANDS.get(capability, ())
    for attempt in range(128):
        role = rng.choice(roles)
        shape = rng.random()
        if capability == "todo_lists":
            name = role
        elif capability == "scripts":
            name = role if shape < 0.65 else f"{area} {rng.choice(('Lights Off', 'Lights On', 'Evening Lighting'))}"
        elif capability == "scenes":
            name = f"{area} {role}"
        elif shape < 0.12:
            name = f"{rng.choice(owners)} {role}"
        elif shape < 0.35 and brands:
            name = f"{area} {rng.choice(brands)} {_ENTITY_TEMPLATES[capability][0]}"
        elif rng.random() < bare_name_rate:
            name = role
        else:
            name = f"{area} {role}"
        if rng.random() < 0.26:
            name = name[:1] + name[1:].lower()
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    for number in range(2, len(taken) + 3):
        name = f"{area} {roles[0]} {number}"
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    raise ValueError("no unique fixture name")


def _capability_slots(size: int, rng: random.Random) -> list[str]:
    counts = {
        "lights": int(size * .40), "switches": int(size * .22),
        "covers": int(size * .10), "fans": int(size * .06), "media_players": int(size * .04),
        "climate": min(2, max(1, size // 24)), "locks": min(3, max(1, size // 24)),
        "scripts": rng.randint(1, 4),
        "vacuums": int(size >= 32), "buttons": int(size >= 32),
        "todo_lists": int(size >= 32), "lawn_mowers": int(size >= 64),
    }
    slots = [capability for capability, count in counts.items() for _ in range(count)]
    slots += ["lights"] * max(0, size - len(slots))
    rng.shuffle(slots)
    return slots[:size]


def _collision_aliases(name: str, area: str, noun: str) -> list[str]:
    if name.casefold().startswith(area.casefold()):
        return [f"{area} {noun}", noun]
    return [noun]


def generate_home(
    index: int,
    size: int,
    rng: random.Random,
    *,
    sayso_entity_area: str | None = None,
    bare_name_rate: float = 0.0,
    holdout: bool | None = None,
) -> dict[str, Any]:
    holdout = _holdout_mode(holdout)
    capabilities = _capability_slots(size, rng)
    entities: list[dict[str, Any]] = []
    if holdout:
        rooms = list(_HOLDOUT_AREAS)
    else:
        rooms = ["Kitchen", "Living Room", "Master Bedroom", "Bathroom", "Hallway", "Entryway", "Garage"]
        extras = [area for area in _AREAS if area not in rooms]
        rooms += rng.sample(extras, min(5, max(1, size // 8)))
    floors = {area: ("Basement" if area == "Basement" else
                     "Upstairs" if "Bedroom" in area or area == "Attic" else "Main Floor") for area in rooms}
    taken: set[str] = set()
    owners = tuple(rng.sample(_HOLDOUT_OWNERS if holdout else _OWNERS, 2))
    nickname_rng = random.Random(f"nicknames:{index}:{size}")
    used_nicknames: set[str] = set()
    for slot, capability in enumerate(capabilities):
        area = rng.choice(device_areas(capability, rooms, holdout=holdout))
        name = _random_entity_name(
            capability, area, slot, index, rng, taken, owners,
            bare_name_rate=bare_name_rate, holdout=holdout,
        )
        taken.add(_slug(name))
        noun = _ENTITY_TEMPLATES[capability][0].lower()
        bare = not name.casefold().startswith(area.casefold())
        if capability in {"scripts", "scenes", "todo_lists"} or bare:
            aliases = [name]
        else:
            aliases = list(dict.fromkeys([name, f"{area} {noun}"]))
        if capability == "lights":
            synonyms = {"light": "lamp", "lights": "lamps", "lamp": "light", "lamps": "lights",
                        "spotlights": "spots", "backlight": "backlighting"}
            alias = re.sub(r"\b(light|lights|lamp|lamps|spotlights|backlight)\b",
                           lambda match: synonyms[match[0].lower()], name, flags=re.I)
            if alias.casefold() != name.casefold():
                aliases.append(alias)
        nickname = _nickname_alias(name, owners, nickname_rng, used_nicknames)
        if nickname:
            aliases.append(nickname)
            used_nicknames.add(nickname.casefold())
        entities.append(make_entity(name=name, capability=capability, area=area,
                                    floor=floors[area], rng=rng, aliases=aliases))

    if size >= 16:
        lights = [entity for entity in entities if entity["capability"] == "lights"]
        if lights:
            base = rng.choice(lights)
            name = _random_entity_name(
                "lights", base["area"], len(entities), index, rng, taken, owners,
                bare_name_rate=bare_name_rate, holdout=holdout,
            )
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="lights",
                    area=base["area"],
                    floor=base["floor"],
                    rng=rng,
                    aliases=_collision_aliases(name, base["area"], "light"),
                )
            )
        tvs = [entity for entity in entities if entity["capability"] == "media_players"]
        if tvs:
            elsewhere = rng.choice([
                a for a in device_areas("media_players", rooms, holdout=holdout) if a != tvs[0]["area"]
            ])
            name = _random_entity_name(
                "media_players", elsewhere, len(entities), index, rng, taken, owners,
                bare_name_rate=bare_name_rate, holdout=holdout,
            )
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="media_players",
                    area=elsewhere,
                    floor=floors[elsewhere],
                    rng=rng,
                    aliases=_collision_aliases(name, elsewhere, "tv"),
                )
            )

    remove_canonical_alias_collisions(entities, avoid_eval_names=True)
    area = sayso_entity_area or entities[index % len(entities)]["area"]
    satellite_rng = random.Random(zlib.crc32(f"satellite:{index}:{size}".encode()))
    outdoor = _HOLDOUT_OUTDOOR_AREAS if holdout else _OUTDOOR_AREAS
    # Satellites are indoor speakers. The draw is taken either way so the RNG stream is unchanged.
    if satellite_rng.random() < SATELLITE_RATE and area not in outdoor:
        brand_name = f"{SATELLITE_NAME_BRAND} {SATELLITE_NAME_SUFFIX}"
        area_name = f"{area} {SATELLITE_NAME_SUFFIX}"
        candidates = [name for name in (brand_name, area_name)
                      if _slug(name) not in taken and name.casefold() not in _eval_entity_names()]
        sat_name = satellite_rng.choice(candidates) if candidates else None
    else:
        sat_name = None
    if sat_name is not None:
        taken.add(_slug(sat_name))
        entities.append(
            make_entity(
                name=sat_name,
                capability="media_players",
                area=area,
                floor=floors[area],
                rng=satellite_rng,
                aliases=[sat_name],
            )
        )
    return {
        "home_id": f"home_{index:06d}_{size}",
        "size": size,
        "sayso_entity_area": area,
        "entities": entities,
        "active_timers": _synthetic_timers(rng, holdout) if rng.random() < 0.3 else [],
        "areas": rooms, "area_floors": floors, "owners": owners,
        "exposure_source": "synthetic",
        "context_entities": _context_entities_for(entities, rooms, index, size, rng.getstate()),
    }



def remove_canonical_alias_collisions(
    entities: list[dict[str, Any]], *, avoid_eval_names: bool = False
) -> None:
    # Eval-name avoidance is for synthetic generation only. Real homes reach this function too, and
    # their fixture names (e.g. 'TV') are eval names that must survive unchanged.
    canonical = {entity["name"].casefold() for entity in entities}
    eval_names = _eval_entity_names() if avoid_eval_names else frozenset()
    for entity in entities:
        other_names = canonical - {entity["name"].casefold()}
        entity["aliases"] = [alias for alias in entity.get("aliases", [])
                             if alias.casefold() not in other_names
                             and alias.casefold() not in eval_names]


_TIMER_NAMES = (
    "Pizza Timer", "Laundry Timer", "Kids Bath Timer", "Pasta Timer", "Oven Timer", "Tea Timer",
    "Egg Timer", "Nap Timer", "Homework Timer",
)


def _synthetic_timers(rng: random.Random, holdout: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "name": rng.choice(_TIMER_NAMES),
            "remaining_seconds": rng.randint(60, 1800),
            "area": rng.choice(_HOLDOUT_AREAS if holdout else _AREAS),
        }
        for _ in range(rng.randint(1, 2))
    ]


def entities_of_capability(home: dict[str, Any], capability: str) -> list[dict[str, Any]]:
    return [e for e in home["entities"] if e.get("capability") == capability]


def entities_in_area(home: dict[str, Any], capability: str, area: str) -> list[dict[str, Any]]:
    return [e for e in entities_of_capability(home, capability) if e["area"] == area]


_TV_ROLE_WORD = re.compile(r"\b(?:t[ .]*v|television)s?\b", re.I)
_TV_MEDIA_ROLES = tuple(role for role in _ROLES["media_players"] if _TV_ROLE_WORD.search(role))
_NONTV_MEDIA_ROLES = tuple(role for role in _ROLES["media_players"] if role not in _TV_MEDIA_ROLES)


def _visible_tv_entities(home: dict[str, Any], area: str) -> list[dict[str, Any]]:
    from generators.gold import refers_to

    return [
        entity
        for entity in entities_in_area(home, "media_players", area)
        if refers_to(entity, "tv") or refers_to(entity, "television")
    ]


def _unique_media_name(area: str, rng: random.Random, taken: set[str], roles: tuple[str, ...]) -> str:
    brands = _BRANDS.get("media_players", ())
    for _ in range(128):
        if roles is _TV_MEDIA_ROLES and brands and rng.random() < 0.35:
            name = f"{area} {rng.choice(brands)} TV"
        else:
            name = f"{area} {rng.choice(roles)}"
        if rng.random() < 0.26:
            name = name[:1] + name[1:].lower()
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    for number in range(2, len(taken) + 3):
        name = f"{area} {roles[0]} {number}"
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    raise ValueError("no unique media player name")


def _append_media_player(
    home: dict[str, Any],
    *,
    area: str,
    floor: str,
    operation: str,
    rng: random.Random,
    taken: set[str],
    tv: bool,
) -> None:
    roles = _TV_MEDIA_ROLES if tv else _NONTV_MEDIA_ROLES
    name = _unique_media_name(area, rng, taken, roles)
    taken.add(_slug(name))
    needed = set(_ENTITY_TEMPLATES["media_players"][2]) | set(required_features("media_players", operation))
    home["entities"].append(
        make_entity(
            name=name,
            capability="media_players",
            area=area,
            floor=floor,
            rng=rng,
            features=tuple(sorted(needed)),
        )
    )


def prepare_media_tv_ambiguity(
    home: dict[str, Any],
    operation: str,
    area: str,
    rng: random.Random,
    index: int,
) -> None:
    if home.get("home_id") == "ambiguity_regression":
        return
    bucket = random.Random(zlib.crc32(f"media-tv-amb:{index}:{area}".encode()))
    roll = bucket.random()
    if roll < 0.85:
        target = 2
    elif roll < 0.90:
        target = 0
    else:
        target = 1
    floor = (home.get("area_floors") or {}).get(area, "Main Floor")
    taken = {entity["entity_id"].split(".", 1)[1] for entity in home["entities"]}
    tvs = _visible_tv_entities(home, area)
    while len(tvs) > target:
        home["entities"].remove(tvs.pop())
    while len(_visible_tv_entities(home, area)) < target:
        _append_media_player(
            home, area=area, floor=floor, operation=operation, rng=rng, taken=taken, tv=True,
        )
    if target >= 2 and bucket.random() < 0.55:
        speakers = [
            entity
            for entity in entities_in_area(home, "media_players", area)
            if entity not in _visible_tv_entities(home, area)
        ]
        if not speakers:
            _append_media_player(
                home, area=area, floor=floor, operation=operation, rng=rng, taken=taken, tv=False,
            )
    remove_canonical_alias_collisions(home["entities"])
