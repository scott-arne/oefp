// swig/oefp.i
// SWIG interface file for oefp Python bindings
%module _oefp

%{
#include "oefp/oefp.h"
#include "oefp/descriptor_source.h"
#include "oefp/descriptor_calculator.h"
#include "oefp/rdkit_descriptors.h"
#include "oefp/atom_descriptor.h"
#include "oefp/kallisto_descriptors.h"
#include <oechem.h>
#include <oegrid.h>
%}

// ============================================================================
// Forward declarations for cross-module SWIG type resolution
// ============================================================================
// These enable typemaps for OpenEye types whose definitions live in the
// OpenEye SWIG runtime (v4). Only types you actually use in your wrapped API
// need full #include — forward declarations suffice for the typemaps.

namespace OEChem {
    class OEMolBase;
    class OEMCMolBase;
    class OEMol;
    class OEGraphMol;
    class OEAtomBase;
    class OEBondBase;
    class OEConfBase;
    class OEMatchBase;
    class OEMolDatabase;
    class oemolistream;
    class oemolostream;
    class OEQMol;
    class OEResidue;
    class OEUniMolecularRxn;
}

namespace OEBio {
    class OEDesignUnit;
    class OEHierView;
    class OEHierResidue;
    class OEHierFragment;
    class OEHierChain;
    class OEInteractionHint;
    class OEInteractionHintContainer;
}

namespace OEDocking {
    class OEReceptor;
}

namespace OEPlatform {
    class oeifstream;
    class oeofstream;
    class oeisstream;
    class oeosstream;
}

namespace OESystem {
    class OEScalarGrid;
    class OERecord;
    class OEMolRecord;
}

namespace OEGraphSim {
    class OEFingerPrint;
}

// ============================================================================
// Cross-runtime SWIG compatibility layer
// ============================================================================
// OpenEye's Python bindings use SWIG runtime v4; our module uses v5.
// Since the runtimes are separate, SWIG_TypeQuery cannot access OpenEye types.
// We use Python isinstance for type safety and directly extract the void*
// pointer from the SwigPyObject struct layout (stable across SWIG versions).
//
// This approach enables passing OpenEye objects between Python and C++ without
// serialization. The macros below generate the boilerplate for each type.

%{
// Minimal SwigPyObject layout compatible across SWIG runtime versions.
// The actual struct may have more fields, but ptr is always first after
// PyObject_HEAD.
struct _SwigPyObjectCompat {
    PyObject_HEAD
    void *ptr;
};

static void* _oefp_extract_swig_ptr(PyObject* obj) {
    PyObject* thisAttr = PyObject_GetAttrString(obj, "this");
    if (!thisAttr) {
        PyErr_Clear();
        return NULL;
    }
    void* ptr = ((_SwigPyObjectCompat*)thisAttr)->ptr;
    Py_DECREF(thisAttr);
    return ptr;
}

// ---- Type checker generator macro ----
// Generates a cached isinstance checker for an OpenEye Python type.
// TAG:    identifier suffix (e.g., oemolbase)
// MODULE: Python module string (e.g., "openeye.oechem")
// CLASS:  Python class name string (e.g., "OEMolBase")
#define DEFINE_OE_TYPE_CHECKER(TAG, MODULE, CLASS) \
    static PyObject* _oefp_oe_##TAG##_type = NULL; \
    static bool _oefp_is_##TAG(PyObject* obj) { \
        if (!_oefp_oe_##TAG##_type) { \
            PyObject* mod = PyImport_ImportModule(MODULE); \
            if (mod) { \
                _oefp_oe_##TAG##_type = PyObject_GetAttrString(mod, CLASS); \
                Py_DECREF(mod); \
            } \
            if (!_oefp_oe_##TAG##_type) return false; \
        } \
        return PyObject_IsInstance(obj, _oefp_oe_##TAG##_type) == 1; \
    }

// ---- Molecule types (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oemolbase,    "openeye.oechem", "OEMolBase")
DEFINE_OE_TYPE_CHECKER(oemcmolbase,  "openeye.oechem", "OEMCMolBase")
DEFINE_OE_TYPE_CHECKER(oemol,        "openeye.oechem", "OEMol")
DEFINE_OE_TYPE_CHECKER(oegraphmol,   "openeye.oechem", "OEGraphMol")
DEFINE_OE_TYPE_CHECKER(oeqmol,       "openeye.oechem", "OEQMol")

// ---- Atom / bond / conformer / residue (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeatombase,   "openeye.oechem", "OEAtomBase")
DEFINE_OE_TYPE_CHECKER(oebondbase,   "openeye.oechem", "OEBondBase")
DEFINE_OE_TYPE_CHECKER(oeconfbase,   "openeye.oechem", "OEConfBase")
DEFINE_OE_TYPE_CHECKER(oeresidue,    "openeye.oechem", "OEResidue")
DEFINE_OE_TYPE_CHECKER(oematchbase,  "openeye.oechem", "OEMatchBase")

// ---- Molecule I/O (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oemolistream, "openeye.oechem", "oemolistream")
DEFINE_OE_TYPE_CHECKER(oemolostream, "openeye.oechem", "oemolostream")
DEFINE_OE_TYPE_CHECKER(oemoldatabase,"openeye.oechem", "OEMolDatabase")

// ---- Reactions (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeunimolecularrxn, "openeye.oechem", "OEUniMolecularRxn")

// ---- Platform streams (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oeifstream,   "openeye.oechem", "oeifstream")
DEFINE_OE_TYPE_CHECKER(oeofstream,   "openeye.oechem", "oeofstream")
DEFINE_OE_TYPE_CHECKER(oeisstream,   "openeye.oechem", "oeisstream")
DEFINE_OE_TYPE_CHECKER(oeosstream,   "openeye.oechem", "oeosstream")

// ---- Records (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oerecord,     "openeye.oechem", "OERecord")
DEFINE_OE_TYPE_CHECKER(oemolrecord,  "openeye.oechem", "OEMolRecord")

// ---- Bio / hierarchy (openeye.oechem) ----
DEFINE_OE_TYPE_CHECKER(oedesignunit, "openeye.oechem", "OEDesignUnit")
DEFINE_OE_TYPE_CHECKER(oehierview,   "openeye.oechem", "OEHierView")
DEFINE_OE_TYPE_CHECKER(oehierresidue,"openeye.oechem", "OEHierResidue")
DEFINE_OE_TYPE_CHECKER(oehierfragment,"openeye.oechem","OEHierFragment")
DEFINE_OE_TYPE_CHECKER(oehierchain,  "openeye.oechem", "OEHierChain")
DEFINE_OE_TYPE_CHECKER(oeinteractionhint,          "openeye.oechem", "OEInteractionHint")
DEFINE_OE_TYPE_CHECKER(oeinteractionhintcontainer, "openeye.oechem", "OEInteractionHintContainer")

// ---- Grid (openeye.oegrid) ----
DEFINE_OE_TYPE_CHECKER(oescalargrid, "openeye.oegrid", "OEScalarGrid")

// ---- Docking (openeye.oedocking) ----
DEFINE_OE_TYPE_CHECKER(oereceptor,   "openeye.oedocking", "OEReceptor")

// ---- Fingerprints (openeye.oegraphsim) ----
DEFINE_OE_TYPE_CHECKER(oefingerprint, "openeye.oegraphsim", "OEFingerPrint")

#undef DEFINE_OE_TYPE_CHECKER

// ---- OEScalarGrid return-type helper (zero-copy pointer swap) ----
static PyObject* _oefp_wrap_as_oe_grid(OESystem::OEScalarGrid* grid) {
    if (!grid) {
        Py_RETURN_NONE;
    }
    PyObject* oegrid_mod = PyImport_ImportModule("openeye.oegrid");
    if (!oegrid_mod) {
        delete grid;
        return NULL;
    }
    PyObject* grid_cls = PyObject_GetAttrString(oegrid_mod, "OEScalarGrid");
    Py_DECREF(oegrid_mod);
    if (!grid_cls) {
        delete grid;
        return NULL;
    }
    PyObject* oe_grid = PyObject_CallNoArgs(grid_cls);
    Py_DECREF(grid_cls);
    if (!oe_grid) {
        delete grid;
        return NULL;
    }
    PyObject* thisAttr = PyObject_GetAttrString(oe_grid, "this");
    if (!thisAttr) {
        PyErr_Clear();
        Py_DECREF(oe_grid);
        delete grid;
        return NULL;
    }
    _SwigPyObjectCompat* swig_this = (_SwigPyObjectCompat*)thisAttr;
    delete reinterpret_cast<OESystem::OEScalarGrid*>(swig_this->ptr);
    swig_this->ptr = grid;
    Py_DECREF(thisAttr);
    return oe_grid;
}

static PyObject* _oefp_wrap_as_oe_fingerprint(OEGraphSim::OEFingerPrint* fp) {
    if (!fp) {
        Py_RETURN_NONE;
    }
    PyObject* oegraphsim_mod = PyImport_ImportModule("openeye.oegraphsim");
    if (!oegraphsim_mod) {
        delete fp;
        return NULL;
    }
    PyObject* fp_cls = PyObject_GetAttrString(oegraphsim_mod, "OEFingerPrint");
    Py_DECREF(oegraphsim_mod);
    if (!fp_cls) {
        delete fp;
        return NULL;
    }
    PyObject* oe_fp = PyObject_CallNoArgs(fp_cls);
    Py_DECREF(fp_cls);
    if (!oe_fp) {
        delete fp;
        return NULL;
    }
    PyObject* thisAttr = PyObject_GetAttrString(oe_fp, "this");
    if (!thisAttr) {
        PyErr_Clear();
        Py_DECREF(oe_fp);
        delete fp;
        return NULL;
    }
    _SwigPyObjectCompat* swig_this = (_SwigPyObjectCompat*)thisAttr;
    delete reinterpret_cast<OEGraphSim::OEFingerPrint*>(swig_this->ptr);
    swig_this->ptr = fp;
    Py_DECREF(thisAttr);
    return oe_fp;
}
%}

// ============================================================================
// Typemap generator macros
// ============================================================================

// Generate const-ref and non-const-ref typemaps for a cross-runtime OpenEye type.
// CPP_TYPE: fully qualified C++ type (e.g., OEChem::OEMolBase)
// CHECKER:  isinstance checker function name
// ERR_MSG:  error message on type mismatch
%define OE_CROSS_RUNTIME_REF_TYPEMAPS(CPP_TYPE, CHECKER, ERR_MSG)

%typemap(in) const CPP_TYPE& (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        if (CHECKER($input)) {
            argp = _oefp_extract_swig_ptr($input);
            if (argp) res = SWIG_OK;
        }
    }
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    $1 = reinterpret_cast< $1_ltype >(argp);
}

%typemap(typecheck, precedence=10) const CPP_TYPE& {
    void *vptr = 0;
    int res = SWIG_ConvertPtr($input, &vptr, $descriptor, SWIG_POINTER_NO_NULL);
    $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
}

%typemap(in) CPP_TYPE& (void *argp = 0, int res = 0) {
    res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
    if (!SWIG_IsOK(res)) {
        if (CHECKER($input)) {
            argp = _oefp_extract_swig_ptr($input);
            if (argp) res = SWIG_OK;
        }
    }
    if (!SWIG_IsOK(res)) {
        SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
    }
    if (!argp) {
        SWIG_exception_fail(SWIG_NullReferenceError, "Null reference.");
    }
    $1 = reinterpret_cast< $1_ltype >(argp);
}

%typemap(typecheck, precedence=10) CPP_TYPE& {
    void *vptr = 0;
    int res = SWIG_ConvertPtr($input, &vptr, $descriptor, SWIG_POINTER_NO_NULL);
    $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
}

%enddef

// Generate nullable-pointer typemaps (accepts None) for a cross-runtime type.
%define OE_CROSS_RUNTIME_NULLABLE_PTR_TYPEMAPS(CPP_TYPE, CHECKER, ERR_MSG)

%typemap(in) const CPP_TYPE* (void *argp = 0, int res = 0) {
    if ($input == Py_None) {
        $1 = NULL;
    } else {
        res = SWIG_ConvertPtr($input, &argp, $descriptor, 0);
        if (!SWIG_IsOK(res)) {
            if (CHECKER($input)) {
                argp = _oefp_extract_swig_ptr($input);
                if (argp) res = SWIG_OK;
            }
        }
        if (!SWIG_IsOK(res)) {
            SWIG_exception_fail(SWIG_ArgError(res), ERR_MSG);
        }
        $1 = reinterpret_cast< $1_ltype >(argp);
    }
}

%typemap(typecheck, precedence=10) const CPP_TYPE* {
    if ($input == Py_None) {
        $1 = 1;
    } else {
        void *vptr = 0;
        int res = SWIG_ConvertPtr($input, &vptr, $descriptor, 0);
        $1 = SWIG_IsOK(res) ? 1 : CHECKER($input) ? 1 : 0;
    }
}

%enddef

// ============================================================================
// Typemap declarations for all OpenEye types
// ============================================================================
// Each type gets const-ref and non-const-ref typemaps. Types that commonly
// appear as optional parameters also get nullable-pointer typemaps.
// These are inert until a wrapped function signature uses the type.

// ---- Molecule hierarchy (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMolBase,    _oefp_is_oemolbase,    "Expected OEMolBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMCMolBase,  _oefp_is_oemcmolbase,  "Expected OEMCMolBase-derived object (OEMCMolBase or OEMol).")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMol,        _oefp_is_oemol,        "Expected OEMol object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEGraphMol,   _oefp_is_oegraphmol,   "Expected OEGraphMol object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEQMol,       _oefp_is_oeqmol,       "Expected OEQMol object.")

// ---- Vector of borrowed molecule pointers (DescriptorCalculator::CalculateBatch) ----
// Accepts a Python sequence of openeye.oechem molecule objects and collects
// borrowed const OEMolBase* pointers into a temporary vector that outlives the
// wrapped call. Each element is converted with the same cross-runtime idiom as
// the per-molecule const OEMolBase& typemap.
//
// CalculateBatch releases the GIL and dereferences these borrowed pointers from
// worker threads, so every referenced Python molecule must stay alive until the
// call returns. A plain list or tuple keeps its own strong reference to each
// element, but PySequence_Check also admits lazy/custom sequences whose
// __getitem__ mints a fresh molecule proxy per call and retains no reference to
// it. For such a sequence the only strong reference is the one PySequence_GetItem
// returns, so dropping it immediately would destroy the proxy (and its C++
// molecule) before CalculateBatch runs. The mols_keepalive local therefore holds
// on to that reference for every converted element, and the matching freearg
// typemap releases them only after $action completes. Holding strong references
// across the call is what makes borrowing the raw pointers safe even for lazy
// sequences.
%typemap(in) const std::vector< const OEChem::OEMolBase* >&
        (std::vector< const OEChem::OEMolBase* > mols_tmp, std::vector< PyObject* > mols_keepalive) {
    if (!PySequence_Check($input)) {
        SWIG_exception_fail(SWIG_TypeError, "Expected a sequence of OEMolBase-derived objects.");
    }
    Py_ssize_t seq_len = PySequence_Size($input);
    if (seq_len < 0) SWIG_fail;
    // A malicious or malformed sequence can report a length far larger than any
    // allocation could satisfy (for example a custom sequence whose __len__
    // returns sys.maxsize). Validate against the vectors' max_size() and guard
    // the reservations so an oversized or failed allocation surfaces as a clean
    // Python exception rather than an uncaught C++ exception that aborts the
    // interpreter. This runs before the element loop while both vectors are
    // empty, so the fail label's freearg has nothing to clean up.
    if (static_cast<std::size_t>(seq_len) > mols_tmp.max_size() ||
        static_cast<std::size_t>(seq_len) > mols_keepalive.max_size()) {
        SWIG_exception_fail(SWIG_ValueError, "Molecule sequence is too large.");
    }
    try {
        mols_tmp.reserve(static_cast<std::size_t>(seq_len));
        mols_keepalive.reserve(static_cast<std::size_t>(seq_len));
    } catch (const std::exception& e) {
        SWIG_exception_fail(SWIG_MemoryError, e.what());
    }
    for (Py_ssize_t idx = 0; idx < seq_len; ++idx) {
        PyObject* item = PySequence_GetItem($input, idx);
        if (!item) {
            for (PyObject* kept : mols_keepalive) Py_DECREF(kept);
            mols_keepalive.clear();
            SWIG_fail;
        }
        void* argp = 0;
        int res = SWIG_ConvertPtr(item, &argp, $descriptor(OEChem::OEMolBase *), 0);
        if (!SWIG_IsOK(res)) {
            if (_oefp_is_oemolbase(item)) {
                argp = _oefp_extract_swig_ptr(item);
                if (argp) res = SWIG_OK;
            }
        }
        if (!SWIG_IsOK(res) || !argp) {
            Py_DECREF(item);
            for (PyObject* kept : mols_keepalive) Py_DECREF(kept);
            mols_keepalive.clear();
            SWIG_exception_fail(SWIG_TypeError, "Expected a sequence of OEMolBase-derived objects.");
        }
        mols_tmp.push_back(reinterpret_cast< const OEChem::OEMolBase* >(argp));
        // Transfer ownership of the reference PySequence_GetItem returned into
        // mols_keepalive; the freearg typemap releases it after the C++ call.
        mols_keepalive.push_back(item);
    }
    $1 = &mols_tmp;
}

// Release the strong references retained by the in typemap. SWIG emits freearg
// after $action on the success path and again at the fail label; the error
// branches above clear mols_keepalive before failing, so a fail-path freearg
// iterates an empty vector and cannot double-DECREF. The local is referenced
// with the $argnum suffix (the SWIG-library convention) so it resolves to the
// same mangled variable the in typemap declared for this argument.
%typemap(freearg) const std::vector< const OEChem::OEMolBase* >& {
    for (PyObject* kept : mols_keepalive$argnum) Py_DECREF(kept);
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_VECTOR) const std::vector< const OEChem::OEMolBase* >& {
    $1 = PySequence_Check($input) ? 1 : 0;
}

// ---- Atom / bond / conformer / residue / match (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEAtomBase,   _oefp_is_oeatombase,   "Expected OEAtomBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEBondBase,   _oefp_is_oebondbase,   "Expected OEBondBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEConfBase,   _oefp_is_oeconfbase,   "Expected OEConfBase-derived object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEResidue,    _oefp_is_oeresidue,    "Expected OEResidue object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMatchBase,  _oefp_is_oematchbase,  "Expected OEMatchBase-derived object.")

// ---- Molecule I/O (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::oemolistream,  _oefp_is_oemolistream, "Expected oemolistream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::oemolostream,  _oefp_is_oemolostream, "Expected oemolostream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEMolDatabase, _oefp_is_oemoldatabase,"Expected OEMolDatabase object.")

// ---- Reactions (OEChem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEChem::OEUniMolecularRxn, _oefp_is_oeunimolecularrxn, "Expected OEUniMolecularRxn object.")

// ---- Platform streams (OEPlatform) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeifstream, _oefp_is_oeifstream, "Expected oeifstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeofstream, _oefp_is_oeofstream, "Expected oeofstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeisstream, _oefp_is_oeisstream, "Expected oeisstream object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEPlatform::oeosstream, _oefp_is_oeosstream, "Expected oeosstream object.")

// ---- Records (OESystem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OERecord,    _oefp_is_oerecord,    "Expected OERecord object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OEMolRecord, _oefp_is_oemolrecord, "Expected OEMolRecord object.")

// ---- Bio / hierarchy (OEBio) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEDesignUnit,   _oefp_is_oedesignunit, "Expected OEDesignUnit object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierView,     _oefp_is_oehierview,   "Expected OEHierView object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierResidue,  _oefp_is_oehierresidue,"Expected OEHierResidue object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierFragment,  _oefp_is_oehierfragment,"Expected OEHierFragment object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEHierChain,    _oefp_is_oehierchain,  "Expected OEHierChain object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEInteractionHint,          _oefp_is_oeinteractionhint,          "Expected OEInteractionHint object.")
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEBio::OEInteractionHintContainer, _oefp_is_oeinteractionhintcontainer, "Expected OEInteractionHintContainer object.")

// ---- Grid (OESystem) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OESystem::OEScalarGrid, _oefp_is_oescalargrid, "Expected OEScalarGrid-derived object.")
OE_CROSS_RUNTIME_NULLABLE_PTR_TYPEMAPS(OESystem::OEScalarGrid, _oefp_is_oescalargrid, "Expected OEScalarGrid or None.")

// OEScalarGrid return-type typemap (wraps C++ grid as native openeye.oegrid object)
%typemap(out) OESystem::OEScalarGrid* {
    $result = _oefp_wrap_as_oe_grid($1);
    if (!$result) SWIG_fail;
}

// ---- Fingerprints (OEGraphSim) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEGraphSim::OEFingerPrint, _oefp_is_oefingerprint, "Expected OEFingerPrint object.")

%typemap(out) OEGraphSim::OEFingerPrint {
    $result = _oefp_wrap_as_oe_fingerprint(new OEGraphSim::OEFingerPrint($1));
    if (!$result) SWIG_fail;
}

// ---- Docking (OEDocking) ----
OE_CROSS_RUNTIME_REF_TYPEMAPS(OEDocking::OEReceptor, _oefp_is_oereceptor, "Expected OEReceptor object.")

// ============================================================================
// Include STL typemaps
// ============================================================================
%include "std_string.i"
%include "std_vector.i"
%include "std_shared_ptr.i"
%include "stdint.i"
%include "exception.i"

// Descriptor sources are handled through shared_ptr so Python can hold and pass
// them (and their derived types) by value into DescriptorSourceEntry, which
// stores a std::shared_ptr<const DescriptorSource>. Declared before the source
// header is %include'd, as SWIG requires.
%shared_ptr(OEFP::DescriptorSource)
%shared_ptr(OEFP::MordredDescriptorSource)
%shared_ptr(OEFP::OpenEyePropertyDescriptorSource)
%shared_ptr(OEFP::RDKitDescriptorSource)

// SWIG's python/std_common.i registers numeric "value" traits
// (swig::traits<T> with value_category plus traits_from / traits_asval) only
// for the *fundamental* primitive spellings (long long, unsigned long long,
// unsigned char, double, unsigned int, ...), keyed by a fragment named
// SWIG_Traits_frag(<spelling>). When std::vector is instantiated over a
// *typedef* spelling such as std::int64_t or std::uint64_t, std_vector.i's
// %traits_swigtype fallback instead registers a pointer_category traits
// fragment under the typedef's own name, so vector elements marshal to Python
// as SwigPyObject pointer proxies (and leak) rather than as int values.
// UInt32Vector / DoubleVector / StringVector work only because their %template
// spelling already matches a fundamental value-traits fragment.
//
// Pre-register value traits for the exact typedef spellings *before* the
// %template declarations. Fragment names are unique, so the earlier definition
// wins over std_vector.i's pointer fallback. Conversions delegate to the
// fundamental SWIG_From / SWIG_AsVal helpers via a lossless cast, which keeps
// them correct whether std::int64_t maps to long or long long on a platform.
%fragment(SWIG_Traits_frag(std::int64_t), "header",
          fragment=SWIG_From_frag(long long),
          fragment=SWIG_AsVal_frag(long long),
          fragment="StdTraits") {
namespace swig {
  template <> struct traits< std::int64_t > {
    typedef value_category category;
    static const char* type_name() { return "std::int64_t"; }
  };
  template <> struct traits_asval< std::int64_t > {
    typedef std::int64_t value_type;
    static int asval(PyObject *obj, value_type *val) {
      long long v = 0;
      int res = SWIG_AsVal(long long)(obj, &v);
      if (SWIG_IsOK(res) && val) *val = static_cast< value_type >(v);
      return res;
    }
  };
  template <> struct traits_from< std::int64_t > {
    typedef std::int64_t value_type;
    static PyObject *from(const value_type& val) {
      return SWIG_From(long long)(static_cast< long long >(val));
    }
  };
}
}

%fragment(SWIG_Traits_frag(std::uint64_t), "header",
          fragment=SWIG_From_frag(unsigned long long),
          fragment=SWIG_AsVal_frag(unsigned long long),
          fragment="StdTraits") {
namespace swig {
  template <> struct traits< std::uint64_t > {
    typedef value_category category;
    static const char* type_name() { return "std::uint64_t"; }
  };
  template <> struct traits_asval< std::uint64_t > {
    typedef std::uint64_t value_type;
    static int asval(PyObject *obj, value_type *val) {
      unsigned long long v = 0;
      int res = SWIG_AsVal(unsigned long long)(obj, &v);
      if (SWIG_IsOK(res) && val) *val = static_cast< value_type >(v);
      return res;
    }
  };
  template <> struct traits_from< std::uint64_t > {
    typedef std::uint64_t value_type;
    static PyObject *from(const value_type& val) {
      return SWIG_From(unsigned long long)(static_cast< unsigned long long >(val));
    }
  };
}
}

namespace std {
%template(StringVector) vector< std::string >;
%template(UInt32Vector) vector< unsigned int >;
%template(Int64Vector) vector< std::int64_t >;
%template(UInt64Vector) vector< std::uint64_t >;
%template(DoubleVector) vector< double >;
// std::uint8_t is unsigned char on every platform, for which SWIG already
// registers fundamental value traits, so this template marshals element
// values (DescriptorBatch::BoolColumn / ColumnValidity return
// std::vector<std::uint8_t>).
%template(UInt8Vector) vector< unsigned char >;
}

// ============================================================================
// Exception handling
// ============================================================================
%exception {
    try {
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (...) {
        SWIG_exception(SWIG_RuntimeError, "Unknown C++ exception");
    }
}

// ============================================================================
// Version macros
// ============================================================================
#define OEFP_VERSION_MAJOR 0
#define OEFP_VERSION_MINOR 2
#define OEFP_VERSION_PATCH 12

// ============================================================================
// Wrapped API
// ============================================================================
%rename(_NativeOEFP) OEFP::OEFP;
%rename(_NativeOEFPBatch) OEFP::OEFPBatch;
%rename(_NativeOEFPCountBatch) OEFP::OEFPCountBatch;
%rename(_NativeOEFPSparseBatch) OEFP::OEFPSparseBatch;
%rename(_NativeDescriptorSet) OEFP::DescriptorSet;
%rename(_NativeDescriptorBatch) OEFP::DescriptorBatch;
%rename(_NativeDescriptorCalculator) OEFP::DescriptorCalculator;
%rename(_NativeMetric) OEFP::Metric;
%rename(_NativeAtomPairGenerator) OEFP::AtomPairGenerator;
%rename(_NativeMorganGenerator) OEFP::MorganGenerator;
%rename(_NativeTopologicalTorsionsGenerator) OEFP::TopologicalTorsionsGenerator;
%rename(_ProfileAtomPairFingerprintStages) OEFP::ProfileAtomPairFingerprint;
%rename(_ProfileMorganFingerprintStages) OEFP::ProfileMorganFingerprint;

%ignore OEFP::OEFPMappingSet::EnvironmentsForBit;
%ignore OEFP::DescriptorSchema::DescriptorSchema;
%ignore OEFP::DescriptorSchema::Definitions;
%ignore OEFP::DescriptorSet::DescriptorSet;
%ignore OEFP::DescriptorSet::Values;
%ignore OEFP::ToArrowRecordBatch;
%ignore OEFP::FromArrowRecordBatch;
%ignore OEFP::WriteDescriptorIpc;
%ignore OEFP::ReadDescriptorIpc;
%ignore OEFP::WriteDescriptorParquet;
%ignore OEFP::ReadDescriptorParquet;

%define OEFP_GIL_RELEASE_EXCEPTION(FUNC)
%exception FUNC {
    PyThreadState* _oefp_thread_state = PyEval_SaveThread();
    try {
        $action
        PyEval_RestoreThread(_oefp_thread_state);
    } catch (const std::exception& e) {
        PyEval_RestoreThread(_oefp_thread_state);
        SWIG_exception(SWIG_RuntimeError, e.what());
    } catch (...) {
        PyEval_RestoreThread(_oefp_thread_state);
        SWIG_exception(SWIG_RuntimeError, "Unknown C++ exception");
    }
}
%enddef

OEFP_GIL_RELEASE_EXCEPTION(OEFP::Compare)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::CDist)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::PDist)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::CompareIntoAddress)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::CDistIntoAddress)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::PDistIntoAddress)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeAtomPairFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeAtomPairCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeAtomPairSparseFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeAtomPairSparseCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeAtomPairDescriptors)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::AtomPairGenerator::Fingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::ProfileAtomPairFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeTopologicalTorsionsFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeTopologicalTorsionsCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeTopologicalTorsionsSparseFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeTopologicalTorsionsSparseCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeTopologicalTorsionsDescriptors)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::TopologicalTorsionsGenerator::Fingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganFingerprintWithMapping)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganCountFingerprintWithMapping)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganSparseCountFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganSparseCountFingerprintWithMapping)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganDescriptors)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganSparseFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMorganSparseFingerprintWithMapping)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MorganGenerator::Fingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::ProfileMorganFingerprint)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeMordredDescriptors)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::MakeRDKitDescriptors)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::DescriptorCalculator::Compute)
OEFP_GIL_RELEASE_EXCEPTION(OEFP::DescriptorCalculator::CalculateBatch)

%include "oefp/fingerprint.h"
%include "oefp/count.h"
%include "oefp/sparse.h"
%include "oefp/annotation.h"
%include "oefp/descriptor_schema.h"
%include "oefp/descriptor_value.h"
%include "oefp/descriptor_selection.h"
%include "oefp/descriptor.h"
%include "oefp/descriptor_batch.h"

// DescriptorSourceEntry has no default constructor, so suppress the sized
// vector constructor and resize(size) wrappers std_vector.i would otherwise
// generate (they require a default-constructible element type).
%std_nodefconst_type(::OEFP::DescriptorSourceEntry)

// The DescriptorCalculator(std::vector<DescriptorSourceEntry>) constructor
// writes its element type unqualified. Because a class named OEFP::OEFP shares
// its name with the enclosing OEFP namespace, SWIG mis-renders that unqualified
// name as OEFP::OEFP::DescriptorSourceEntry in the generated wrapper, which does
// not compile. Suppress the header constructor and re-expose it with a fully
// qualified vector parameter that unifies with the DescriptorSourceEntryVector
// template below.
%ignore OEFP::DescriptorCalculator::DescriptorCalculator;
// Ignore the context/request Compute overloads; Python only needs the
// single-molecule overload (the source classes route internally).
%ignore OEFP::DescriptorSource::Compute(const OEChem::OEMolBase&, ComputeContext&, const ColumnRequest&) const;
%ignore OEFP::DescriptorSource::Compute(const OEChem::OEMolBase&, ComputeContext&) const;
// Ignore the context/request overloads of descriptor-making functions; only
// the single-molecule overload is needed in Python (the source classes handle
// the context/request routing internally).
%ignore OEFP::MakeMordredDescriptors(const OEChem::OEMolBase&, ComputeContext&, const ColumnRequest&);
%ignore OEFP::MakeRDKitDescriptors(const OEChem::OEMolBase&, ComputeContext&, const ColumnRequest&);
%include "oefp/descriptor_source.h"
%include "oefp/descriptor_calculator.h"

// A class named OEFP::OEFP shares its name with the enclosing OEFP namespace.
// This confuses SWIG 4.4.1's scoped-name resolution inside the namespace: it
// suppresses the synthesized shared_ptr<Derived> -> shared_ptr<const Base>
// upcast that %shared_ptr would normally generate, so a concrete source proxy
// (MordredDescriptorSource / OpenEyePropertyDescriptorSource) cannot be handed
// to the DescriptorSourceEntry constructor that takes a
// std::shared_ptr<const DescriptorSource>. Provide explicit per-concrete-type
// constructor overloads; the leading "::" qualification sidesteps the same
// name-resolution bug, and shared_ptr<Derived> converts to
// shared_ptr<const Base> by the ordinary C++ conversion.
%extend OEFP::DescriptorSourceEntry {
    DescriptorSourceEntry(std::shared_ptr< ::OEFP::MordredDescriptorSource > source) {
        return new OEFP::DescriptorSourceEntry(std::move(source));
    }
    DescriptorSourceEntry(
        std::shared_ptr< ::OEFP::MordredDescriptorSource > source,
        ::OEFP::DescriptorSelection selection) {
        return new OEFP::DescriptorSourceEntry(std::move(source), std::move(selection));
    }
    DescriptorSourceEntry(std::shared_ptr< ::OEFP::OpenEyePropertyDescriptorSource > source) {
        return new OEFP::DescriptorSourceEntry(std::move(source));
    }
    DescriptorSourceEntry(
        std::shared_ptr< ::OEFP::OpenEyePropertyDescriptorSource > source,
        ::OEFP::DescriptorSelection selection) {
        return new OEFP::DescriptorSourceEntry(std::move(source), std::move(selection));
    }
    DescriptorSourceEntry(std::shared_ptr< ::OEFP::RDKitDescriptorSource > source) {
        return new OEFP::DescriptorSourceEntry(std::move(source));
    }
    DescriptorSourceEntry(
        std::shared_ptr< ::OEFP::RDKitDescriptorSource > source,
        ::OEFP::DescriptorSelection selection) {
        return new OEFP::DescriptorSourceEntry(std::move(source), std::move(selection));
    }
}
namespace std {
%template(DescriptorSourceEntryVector) vector< ::OEFP::DescriptorSourceEntry >;
}
// Cancel the constructor ignore set above so the %extend replacement below is
// wrapped. The ignore only needed to keep SWIG from generating the header's
// unqualified-parameter constructor (which mis-renders as OEFP::OEFP::...).
%rename("%s") OEFP::DescriptorCalculator::DescriptorCalculator;
%extend OEFP::DescriptorCalculator {
    DescriptorCalculator(const std::vector< ::OEFP::DescriptorSourceEntry >& entries) {
        return new OEFP::DescriptorCalculator(entries);
    }
}

%include "oefp/descriptor_arrow.h"
%include "oefp/atom_pair.h"
%include "oefp/topological_torsions.h"
%include "oefp/mordred.h"
// rdkit_descriptors.h: SWIG's OEFP::OEFP collision misqualifies return types
// as OEFP::OEFP::DescriptorSchema (breaks compile). Ignore the header's free
// functions and provide inline wrappers in global scope with explicit qualification.
%ignore OEFP::RDKitDescriptorSchema;
%ignore OEFP::MakeRDKitDescriptors;
%include "oefp/rdkit_descriptors.h"
// The GIL-release on OEFP::MakeRDKitDescriptors above (line 686) targets the
// namespaced function, which is %ignore'd and never wrapped. Apply the GIL macro
// to the global trampoline (MakeRDKitDescriptors unqualified) so the EXPORTED
// function releases the GIL during computation. Must appear before %inline.
OEFP_GIL_RELEASE_EXCEPTION(MakeRDKitDescriptors)
%inline %{
std::shared_ptr<const ::OEFP::DescriptorSchema> RDKitDescriptorSchema() {
    return ::OEFP::RDKitDescriptorSchema();
}
::OEFP::DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol) {
    return ::OEFP::MakeRDKitDescriptors(mol);
}
%}

// kallisto_descriptors.h: Same OEFP::OEFP collision pattern as RDKit above.
// Ignore the namespaced free functions and provide global-scope trampolines.
// Expose AtomDescriptorBatch and BondDescriptorBatch (renamed) with minimal methods for Python zero-copy read
%ignore OEFP::AtomDescriptorSet;
%ignore OEFP::BondDescriptorSet;
// Ignore Append methods (internal use only)
%ignore OEFP::AtomDescriptorBatch::Append;
%ignore OEFP::BondDescriptorBatch::Append;
%rename(_NativeAtomDescriptorBatch) OEFP::AtomDescriptorBatch;
%rename(_NativeBondDescriptorBatch) OEFP::BondDescriptorBatch;
%include "oefp/atom_descriptor.h"
// Ignore all kallisto_descriptors.h content except what we expose via trampolines
%ignore OEFP::coordination_numbers;
%ignore OEFP::proximity_shells;
%ignore OEFP::eeq_charges;
%ignore OEFP::polarizabilities;
%ignore OEFP::van_der_waals_radii;
%ignore OEFP::KallistoAtomDescriptorSchema;
%ignore OEFP::KallistoBondDescriptorSchema;
%ignore OEFP::MakeKallistoAtomDescriptorBatch;
%ignore OEFP::MakeKallistoAtomDescriptors;
%ignore OEFP::MakeKallistoBondDescriptors;
%ignore OEFP::KallistoSterimol;
%ignore OEFP::Sterimol;
%ignore OEFP::KallistoAtomDescriptorSource;
%ignore OEFP::KallistoBondDescriptorSource;
%include "oefp/kallisto_descriptors.h"
// Apply GIL-release to the global trampolines (must appear before %inline)
OEFP_GIL_RELEASE_EXCEPTION(MakeKallistoAtomDescriptorBatch)
OEFP_GIL_RELEASE_EXCEPTION(MakeKallistoBondDescriptorBatch)
OEFP_GIL_RELEASE_EXCEPTION(KallistoAtomDescriptorCalculateBatch)
OEFP_GIL_RELEASE_EXCEPTION(KallistoBondDescriptorCalculateBatch)
OEFP_GIL_RELEASE_EXCEPTION(KallistoSterimol)
%inline %{
// Atom descriptors
std::shared_ptr<const ::OEFP::DescriptorSchema> KallistoAtomDescriptorSchema() {
    return ::OEFP::KallistoAtomDescriptorSchema();
}
std::vector<std::string> KallistoAtomColumnNames() {
    auto schema = ::OEFP::KallistoAtomDescriptorSchema();
    std::vector<std::string> names;
    names.reserve(schema->Size());
    for (std::size_t i = 0; i < schema->Size(); ++i) {
        names.push_back(schema->Definition(i).name);
    }
    return names;
}
::OEFP::AtomDescriptorBatch MakeKallistoAtomDescriptorBatch(const OEChem::OEMolBase& mol) {
    return ::OEFP::MakeKallistoAtomDescriptorBatch(mol);
}
::OEFP::AtomDescriptorBatch MakeKallistoAtomDescriptorBatch(const OEChem::OEMolBase& mol, int charge) {
    return ::OEFP::MakeKallistoAtomDescriptorBatch(mol, charge);
}

// Bond descriptors
std::shared_ptr<const ::OEFP::DescriptorSchema> KallistoBondDescriptorSchema() {
    return ::OEFP::KallistoBondDescriptorSchema();
}
std::vector<std::string> KallistoBondColumnNames() {
    auto schema = ::OEFP::KallistoBondDescriptorSchema();
    std::vector<std::string> names;
    names.reserve(schema->Size());
    for (std::size_t i = 0; i < schema->Size(); ++i) {
        names.push_back(schema->Definition(i).name);
    }
    return names;
}
::OEFP::BondDescriptorBatch MakeKallistoBondDescriptorBatch(const OEChem::OEMolBase& mol) {
    ::OEFP::BondDescriptorSet set = ::OEFP::MakeKallistoBondDescriptors(mol);
    auto schema = ::OEFP::KallistoBondDescriptorSchema();
    ::OEFP::BondDescriptorBatch batch = ::OEFP::BondDescriptorBatch::Empty(schema);
    batch.Append(set);
    return batch;
}

// Sterimol for a directed bond (returns a 3-tuple or throws if ineligible)
std::vector<double> KallistoSterimol(const OEChem::OEMolBase& mol, std::size_t origin, std::size_t partner) {
    auto result = ::OEFP::KallistoSterimol(mol, origin, partner);
    if (!result) {
        throw std::invalid_argument("KallistoSterimol: molecule ineligible, indices out of bounds, or vdW computation failed");
    }
    return {result->l, result->b1, result->b5};
}

// Batch calculations (GIL-released)
::OEFP::AtomDescriptorBatch KallistoAtomDescriptorCalculateBatch(
    const std::vector<const OEChem::OEMolBase*>& mols
) {
    ::OEFP::KallistoAtomDescriptorSource source;
    return source.CalculateBatch(mols);
}

::OEFP::AtomDescriptorBatch KallistoAtomDescriptorCalculateBatch(
    const std::vector<const OEChem::OEMolBase*>& mols,
    int charge
) {
    ::OEFP::KallistoAtomDescriptorSource source(charge);
    return source.CalculateBatch(mols);
}

::OEFP::BondDescriptorBatch KallistoBondDescriptorCalculateBatch(
    const std::vector<const OEChem::OEMolBase*>& mols
) {
    ::OEFP::KallistoBondDescriptorSource source;
    return source.CalculateBatch(mols);
}
%}

namespace std {
%template(OEFPVector) vector< ::OEFP::OEFP >;
%template(OEFPCountVector) vector< ::OEFP::OEFPCount >;
%template(OEFPSparseVector) vector< ::OEFP::OEFPSparse >;
%template(DescriptorSetVector) vector< ::OEFP::DescriptorSet >;
}

%include "oefp/batch.h"
%include "oefp/count_batch.h"
%include "oefp/sparse_batch.h"
%include "oefp/morgan.h"
%include "oefp/metric.h"
%include "oefp/compare.h"
%include "oefp/openeye.h"

// ============================================================================
// Module-level Python convenience code
// ============================================================================
%pythoncode %{
__version__ = "0.2.12"
%}
