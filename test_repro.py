import time
import tracemalloc
import carga_tratamento_dados as carga

tracemalloc.start()
start_time = time.time()

df_treino, df_validacao = carga.carregar_dados()

end_time = time.time()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

elapsed_time = end_time - start_time
peak_mem_mb = peak_mem / (1024 * 1024)

print(f"Tamanho dataset treinamento: {len(df_treino)}")
print(f"Tamanho dataset validação: {len(df_validacao)}")
print(df_treino.head())
print(f"--- EXECUCAO CONCLUIDA ---")
print(f"Tempo total de execucao: {elapsed_time:.2f} s")
print(f"Pico de memoria (tracemalloc): {peak_mem_mb:.2f} MB")

