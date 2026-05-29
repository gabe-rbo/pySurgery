def test_decision_key_single_definition():
    from pysurgery.adams import e_infinity_resolver as base
    from pysurgery.adams import interactive_resolver as inter
    from pysurgery.adams import lean_resolver as lean
    assert inter._decision_key is base._decision_key
    assert lean._decision_key is base._decision_key
