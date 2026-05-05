# MCQA Schema

## Normalized Query File

Every subset has `queries/queries.csv` with:

```csv
id,dataset,domain,structure,query,answer,evidence_graphml,evidence_text,source_file
```

## Evidence

Evidence is provided as GraphML subgraphs. When available, a text evidence paragraph is also included.

GraphML nodes represent entities. GraphML edges represent relations or evidence constraints used to derive the query and answer.
