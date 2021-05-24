import pathlib

from digitize import lead, metadata


def testCroppingToDict():
    crop = metadata.CropLocation(0,0,10,10)
    dictionary = crop.toDict()

    expected = {
        'x': 0,
        'y': 0,
        'height': 10,
        'width': 10
    }
    assert dictionary == expected


def testCroppingFromDict():
    dictionary = {
        'x': 0,
        'y': 0,
        'height': 10,
        'width': 10
    }
    crop = metadata.CropLocation.fromDict(dictionary)

    expected = metadata.CropLocation(0,0,10,10)
    assert crop == expected


def testEcgMetadataSerialize():
    myEcg: metadata.EcgMetadata = {
        lead.Lead.I: metadata.LeadMetadata(
            pathlib.Path('./data/SOHSU10121052013140_0001.tif'),
            10,
            25,
            metadata.CropLocation(
                24,
                350,
                203,
                348
            ),
            2500
        ),
        lead.Lead.II: metadata.LeadMetadata(
            pathlib.Path('./data/SOHSU10121052013140_0001.tif'),
            10,
            25
        ),
    }
    serialized = metadata.serializeEcgMetdata(myEcg)

    expected = '{"I": {"start": 2500, "cropping": {"x": 24, "y": 350, "width": 203, "height": 348}}, "II": {}, "file": "./data/SOHSU10121052013140_0001.tif", "timeScale": 10, "voltageScale": 25}'
    # When Metadata is serialized the path becomes absolute so a direct comparison doesn't work
    assert '{"I": {"start": 2500, "cropping": {"x": 24, "y": 350, "width": 203, "height": 348}}, "II": {}, "file": ' in serialized
    assert '/data/SOHSU10121052013140_0001.tif", "timeScale": 10, "voltageScale": 25}' in serialized


def testEcgMetadataDeserialize():
    serialized = '{"I": {"start": 2500, "cropping": {"x": 24, "y": 350, "width": 203, "height": 348}}, "II": {}, "file": "./data/SOHSU10121052013140_0001.tif", "timeScale": 10, "voltageScale": 25}'
    loadedEcg = metadata.deserializeEcgMetdata(serialized)

    expected: metadata.EcgMetadata = {
        lead.Lead.I: metadata.LeadMetadata(
            pathlib.Path('./data/SOHSU10121052013140_0001.tif'),
            10,
            25,
            metadata.CropLocation(
                24,
                350,
                203,
                348
            ),
            2500
        ),
        lead.Lead.II: metadata.LeadMetadata(
            pathlib.Path('./data/SOHSU10121052013140_0001.tif'),
            10,
            25
        ),
    }
    assert loadedEcg == expected

