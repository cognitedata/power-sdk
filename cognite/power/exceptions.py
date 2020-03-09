class WrongPowerTypeError(Exception):
    pass


class SinglePowerAssetExpected(Exception):
    pass


class MixedPowerAssetListException(Exception):
    pass


def assert_single_result(assets, message=None):
    if len(assets) != 1:
        raise SinglePowerAssetExpected(message or "Expected a single asset result, but found {}".format(len(assets)))
    return assets[0]
