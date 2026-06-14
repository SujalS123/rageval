import os
import glob
import re

files = glob.glob('docs/**/*.md', recursive=True) + ['README.md']

def update_file(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        c = f.read()
    orig_c = c
    
    # 1. RAGTracer
    c = c.replace('RAGTracer(judge=judge)', 'RAGTracer(metrics=[])')
    c = c.replace('tracer = RAGTracer(judge=judge)', 'tracer = RAGTracer(metrics=[])')
    
    c = c.replace('tracer.trace(query=', 'tracer.trace(trace_id=')
    
    if 'with tracer.trace(query=' in orig_c or 'with tracer.trace(trace_id=' in c:
        c = re.sub(r'with tracer\.trace\((?:query|trace_id)="([^"]+)"\) as (\w+):', 
                   r'with tracer.trace(trace_id="\1") as \2:\n    \2._current_step.inputs["query"] = "\1"', c)
        c = re.sub(r'with tracer\.trace\((?:query|trace_id)=query\) as (\w+):', 
                   r'with tracer.trace(trace_id="req-123") as \1:\n    \1._current_step.inputs["query"] = query', c)

    # 2. SemanticDriftDetector
    c = c.replace('.fit_baseline(', '.set_baseline(')
    c = c.replace('detector.analyze(', 'detector.detect(')
    c = c.replace('report.new_clusters', 'report.new_topic_clusters')
    
    if 'detector.set_baseline(' in c and 'detector.set_knowledge_base(' not in c:
        c = re.sub(r'(detector\.set_baseline\([^)]+\))', r'\1\n# We must also set the knowledge base for coverage math\ndetector.set_knowledge_base(["Your knowledge base docs..."])', c)
        
    c = re.sub(r"for cluster in report\.new_topic_clusters:\n\s+print\(f\"- \{cluster\['theme'\]\} \(\{cluster\['count'\]\} queries\)\"\)", 
               "print(report.summary())", c)
        
    # 3. Regression Tracker
    c = c.replace('tracker.save_run(name=', 'tracker.save_run(run_name=')
    
    # 4. ConsistencyAnalyzer
    c = c.replace('report.score', 'report.consistency_score')
    c = c.replace('report.details', 'report.inconsistencies')
    
    if 'analyzer.analyze(' in c:
        c = re.sub(r'base_query=("[^"]+")', r'query=\1', c)
        c = re.sub(r'num_paraphrases=\d+', r'paraphrases=["Why did the 2008 financial crisis happen?", "What were the causes of the 2008 recession?"]', c)
    
    # 5. AutoEval
    c = c.replace('@monitor.watch', '@autoeval.monitor')
    c = c.replace('rolling_window=', 'window_size=')
    
    # 6. FailureTaxonomyBuilder
    c = c.replace('cluster.size', 'cluster.count')
    c = c.replace('cluster.root_cause', 'cluster.trigger')
    c = c.replace('cluster.suggested_fix', 'cluster.fix')
    c = c.replace('cluster.example_queries', 'cluster.example_evidence')
    
    if c != orig_c:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(c)
        print("Updated", fp)

for f in files:
    if os.path.exists(f):
        update_file(f)
