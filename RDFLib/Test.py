# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:49:04 2025

@author: Stephenson
"""

import rdflib
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
import requests
import json
from pathlib import Path

def read_rdf_file(file_path, format_type='xml'):
    """
    Read an RDF file using rdflib
    
    Args:
        file_path (str): Path to the RDF file
        format_type (str): Format of the RDF file ('xml', 'turtle', 'n3', 'nt', 'json-ld')
    
    Returns:
        Graph: rdflib Graph object
    """
    g = Graph()
    
    try:
        # Parse the RDF file
        g.parse(file_path, format=format_type)
        print(f"Successfully loaded {len(g)} triples from {file_path}")
        return g
    except Exception as e:
        print(f"Error reading RDF file: {e}")
        return None



def analyze_fibo_ontology(graph):
    """
    Analyze a FIBO ontology graph and extract key information
    """

    # Find all classes
    classes = set()
    for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
        classes.add(s)
    
    # Find all object properties
    object_properties = set()
    for s, p, o in graph.triples((None, RDF.type, OWL.ObjectProperty)):
        object_properties.add(s)
    
    # Find all data properties
    data_properties = set()
    for s, p, o in graph.triples((None, RDF.type, OWL.DatatypeProperty)):
        data_properties.add(s)
    
    return {
        'classes': list(classes),
        'object_properties': list(object_properties),
        'data_properties': list(data_properties),
        'total_triples': len(graph)
    }

# 4. Query RDF data using SPARQL
def query_fibo_lending_concepts(graph):
    """
    Query for lending-related concepts in FIBO
    """
    # SPARQL query to find lending-related classes
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX fibo-fbc: <https://spec.edmcouncil.org/fibo/ontology/FBC/>
    
    SELECT DISTINCT ?class ?label ?comment
    WHERE {
        ?class a owl:Class .
        OPTIONAL { ?class rdfs:label ?label }
        OPTIONAL { ?class rdfs:comment ?comment }
        FILTER(
            CONTAINS(LCASE(STR(?class)), "loan") ||
            CONTAINS(LCASE(STR(?class)), "mortgage") ||
            CONTAINS(LCASE(STR(?class)), "credit") ||
            CONTAINS(LCASE(STR(?class)), "debt") ||
            CONTAINS(LCASE(STR(?label)), "loan") ||
            CONTAINS(LCASE(STR(?label)), "mortgage")
        )
    }
    """
    
    results = graph.query(query)
    return list(results)

# 5. Extract schema information for database modeling
def extract_schema_info(graph):
    """
    Extract schema information that can be used for database modeling
    """
    schema_info = {
        'entities': {},
        'relationships': [],
        'attributes': {}
    }
    
    # Get all classes (potential entities)
    for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
        class_name = str(s).split('/')[-1]  # Get the last part of URI
        
        # Get label and comment
        label = None
        comment = None
        for _, _, label_obj in graph.triples((s, RDFS.label, None)):
            label = str(label_obj)
            break
        
        for _, _, comment_obj in graph.triples((s, RDFS.comment, None)):
            comment = str(comment_obj)
            break
        
        schema_info['entities'][class_name] = {
            'uri': str(s),
            'label': label,
            'comment': comment
        }
    
    # Get object properties (potential relationships)
    for s, p, o in graph.triples((None, RDF.type, OWL.ObjectProperty)):
        prop_name = str(s).split('/')[-1]
        
        # Find domain and range
        domain = None
        range_val = None
        
        for _, _, domain_obj in graph.triples((s, RDFS.domain, None)):
            domain = str(domain_obj).split('/')[-1]
            break
            
        for _, _, range_obj in graph.triples((s, RDFS.range, None)):
            range_val = str(range_obj).split('/')[-1]
            break
        
        schema_info['relationships'].append({
            'property': prop_name,
            'uri': str(s),
            'domain': domain,
            'range': range_val
        })
    
    return schema_info

# 6. Convert to database-friendly format
def generate_sql_schema(schema_info):
    """
    Generate basic SQL CREATE TABLE statements from schema info
    """
    sql_statements = []
    
    for entity_name, entity_info in schema_info['entities'].items():
        # Basic table creation
        sql = f"CREATE TABLE {entity_name} (\n"
        sql += f"    id UUID PRIMARY KEY,\n"
        sql += f"    uri VARCHAR(500) UNIQUE,\n"
        
        if entity_info['label']:
            sql += f"    label VARCHAR(255),\n"
        
        sql += f"    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
        sql += f"    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        sql += ");\n"
        
        # Add comment if available
        if entity_info['comment']:
            sql += f"COMMENT ON TABLE {entity_name} IS '{entity_info['comment']}';\n"
        
        sql_statements.append(sql)
    
    return sql_statements

# 7. Main execution example
def main():
    """
    Example usage of the RDF reading functions
    """
    
    # Example 1: Read local RDF file
    # graph = read_rdf_file('path/to/your/fibo-file.rdf', 'xml')
    
    # Example 2: Read FIBO file(replace with actual FIBO URL)
    graph = read_rdf_file(r'C:\Users\Stephenson\Downloads\Mortgages.rdf')


    
    if graph:
        # Analyze the ontology
        analysis = analyze_fibo_ontology(graph)
        print(f"\nOntology Analysis:")
        print(f"Classes found: {len(analysis['classes'])}")
        print(f"Object Properties: {len(analysis['object_properties'])}")
        print(f"Data Properties: {len(analysis['data_properties'])}")
        print(f"Total triples: {analysis['total_triples']}")
        
        # Query for lending concepts
        lending_concepts = query_fibo_lending_concepts(graph)
        print(f"\nLending-related concepts found: {len(lending_concepts)}")
        
        # Extract schema information
        schema_info = extract_schema_info(graph)
        print(f"\nEntities extracted: {len(schema_info['entities'])}")
        print(f"Relationships extracted: {len(schema_info['relationships'])}")
        
        # Generate SQL schema (first 3 tables as example)
        sql_statements = generate_sql_schema(schema_info)
        print("\nSample SQL Schema:")
        for i, sql in enumerate(sql_statements[:3]):
            print(f"-- Table {i+1}")
            print(sql)
