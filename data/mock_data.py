"""
Mock data for testing before Eden's data pipeline is ready.
Format mirrors what Eden will eventually provide.
"""

MOCK_INPUT = {
    "island_name": "Nishinoshima",
    "sections": ["Geology", "Ecology", "Climate", "History"],  # provided by Eden (from Wikipedia or defaults)
    "chunks": [
        {
            "chunk_id": "chunk_001",
            "text": (
                "Nishinoshima is a small volcanic island located in the Philippine Sea, "
                "part of the Ogasawara Islands chain in Tokyo, Japan. "
                "The island sits approximately 130 km west of Chichi-jima."
            ),
            "source_url": "https://en.wikipedia.org/wiki/Nishinoshima"
        },
        {
            "chunk_id": "chunk_002",
            "text": (
                "The island has grown significantly due to volcanic eruptions since 2013, "
                "expanding its land area from 0.25 km² to over 2.89 km² by 2020."
            ),
            "source_url": "https://en.wikipedia.org/wiki/Nishinoshima"
        },
        {
            "chunk_id": "chunk_003",
            "text": (
                "Nishinoshima is a stratovolcano sitting atop a seamount. "
                "The 2013-2014 eruption produced basaltic andesite lava flows "
                "that merged with the original island."
            ),
            "source_url": "https://www.volcanodiscovery.com/nishinoshima.html"
        },
        {
            "chunk_id": "chunk_004",
            "text": (
                "Before the 2013 eruptions, Nishinoshima hosted colonies of "
                "masked boobies and brown boobies. "
                "The lava flows destroyed most of the existing habitat."
            ),
            "source_url": "https://en.wikipedia.org/wiki/Nishinoshima"
        },
        {
            "chunk_id": "chunk_005",
            "text": (
                "The island has a subtropical oceanic climate with warm temperatures "
                "year-round and significant rainfall, particularly during typhoon season "
                "from June to November."
            ),
            "source_url": "https://www.jma.go.jp/nishinoshima"
        },
        {
            "chunk_id": "chunk_006",
            "text": (
                "Nishinoshima was first officially surveyed by Japanese authorities in 1904. "
                "A previous eruption in 1973-1974 also expanded the island before activity ceased."
            ),
            "source_url": "https://en.wikipedia.org/wiki/Nishinoshima"
        },
    ]
}


def get_mock_input() -> dict:
    """Return mock input data for testing."""
    return MOCK_INPUT