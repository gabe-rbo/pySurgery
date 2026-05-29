def test_single_canonical_class():
    from pysurgery.homotopy.rational_homotopy import RationalHomotopyGroup as A
    from pysurgery.homotopy.higher_homotopy_groups import RationalHomotopyGroup as B
    assert A is B, "RationalHomotopyGroup must resolve to one class"