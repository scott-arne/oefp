"""Guard the native numeric comparison surface exposed to Python.

``swig/oefp.i`` ``%include``s ``oefp/descriptor_compare.h`` wholesale, which would otherwise
export the raw-pointer numeric forms alongside the intended address shims. SWIG converts Python
``None`` to a null pointer for a ``const double*`` parameter, and ``PDistNumeric`` /
``CDistNumeric`` dereference their values pointer inside the kernel, so an exported pointer form
is a segfault reachable from pure Python. ``%ignore`` directives keep them off the module; these
tests lock that in.
"""

from oefp import _native

POINTER_FORMS = (
    "PDistNumeric",
    "PDistNumericInto",
    "CDistNumeric",
    "CDistNumericInto",
)

ADDRESS_FORMS = (
    "PDistNumericAddress",
    "PDistNumericIntoAddress",
    "CDistNumericAddress",
    "CDistNumericIntoAddress",
    "ColumnStatisticsAddress",
    "CovarianceMatrixAddress",
    "InverseCovarianceMatrixAddress",
)


def test_raw_pointer_numeric_forms_are_not_exported():
    exported = [name for name in POINTER_FORMS if hasattr(_native, name)]
    assert exported == [], f"raw-pointer numeric forms reachable from Python: {exported}"


def test_address_forms_are_still_exported():
    missing = [name for name in ADDRESS_FORMS if not hasattr(_native, name)]
    assert missing == [], f"address forms missing from the native module: {missing}"


def test_batch_statistics_overloads_survive_the_ignores():
    # The pointer overloads of these three are null-safe, so they are deliberately NOT ignored;
    # a name-level ignore would take the DescriptorBatch overloads Task 12 needs with them.
    for name in ("ColumnStatistics", "CovarianceMatrix", "InverseCovarianceMatrix"):
        assert hasattr(_native, name), f"{name} disappeared from the native module"
