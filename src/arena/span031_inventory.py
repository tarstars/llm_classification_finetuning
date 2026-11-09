"""Resolve the known rejected-save draft through the complete owned catalog."""
from types import SimpleNamespace
from unittest.mock import patch

from .experiment import atomic_json,utc_now
from .span004_support import inventory as legacy_inventory,read_call


def resolve_profile(profile,owned,expected_ref):
    owner=expected_ref.split('/')[0].lower()
    refs=[getattr(x,'ref','') for x in profile if getattr(x,'ref','')]
    assert len(refs)==len(set(refs)),'Profile repeats a notebook'
    missing=[x for x in profile if not getattr(x,'ref','')]
    assert len(missing)==1,'Only the specifically recovered draft is understood'
    stub=missing[0]
    assert stub.title=='[Private Notebook]' and str(stub.last_run_time)[:10]=='2010-04-01'
    assert not any(getattr(stub,k,None) for k in ['id','author','slug','current_version_number'])
    canonical=[x.ref for x in owned]
    assert len(canonical)==len(set(canonical))==len(profile)
    assert all(x.ref and x.owner.lower()==owner for x in owned),'Owned catalog identity is incomplete'
    assert all(ref.lower().startswith(owner+'/') for ref in refs)
    assert set(refs)<=set(canonical) and set(canonical)-set(refs)=={expected_ref}
    resolved=[x if getattr(x,'ref','') else SimpleNamespace(ref=expected_ref,last_run_time=stub.last_run_time) for x in profile]
    return resolved,dict(profile_count=len(profile),addressable_before=len(refs),owned_catalog_count=len(canonical),
        recovered_owned_reference=expected_ref,all_existing_references_preserved=True,placeholder_dropped=False)


def inventory_with_resolution(api,start,expected_ref,proof_path):
    response=read_call(lambda:api.kernels_list_with_response(mine=True,page_size=100,sort_by='dateRun'))
    profile=list(response.kernels or [])
    assert not response.next_page_token and len(profile)<100,'Catalog is incomplete; do not truncate it'
    resolution=None
    if any(not getattr(x,'ref','') for x in profile):
        from kaggle.api.kaggle_api_extended import ListEntitiesRequest,ListEntitiesFilters,ApiListType
        request=ListEntitiesRequest();api._set_paging(request,100,None)
        request.canonical_order_by=api._resolve_search_sort_by(None)
        filters=ListEntitiesFilters();filters.query=''
        filters.document_types=api._resolve_search_document_types(['notebook'])
        filters.list_type=ApiListType.API_LIST_TYPE_YOUR_WORK;request.filters=filters
        with api.build_kaggle_client() as client:
            search=read_call(lambda:client.search.search_api_client.list_entities(request))
        assert not search.next_page_token,'Owned catalog is incomplete; paginate before admission'
        owned=[api._search_document_to_row(d) for d in (search.documents or [])]
        profile,resolution=resolve_profile(profile,owned,expected_ref)
        # Source access to the private draft is unnecessary. Its run status is
        # directly observable; a run must never be silently discarded.
        try:read_call(lambda:api.kernels_status(expected_ref))
        except Exception as error:
            assert getattr(getattr(error,'response',None),'status_code',None)==404,'Draft run state unresolved'
        else:raise AssertionError('Rejected draft now has a run: observe it instead of launching')
        resolution.update(at=utc_now(),profile_complete=True,owned_catalog_complete=True,
            directly_observed_no_run_http_status=404,private_source_access_required=False)
    with patch.object(api,'kernels_list',return_value=profile):
        result=legacy_inventory(api,start)
    if resolution:
        result['owned_identity_resolution']=resolution
        atomic_json(proof_path,resolution)
    return result
