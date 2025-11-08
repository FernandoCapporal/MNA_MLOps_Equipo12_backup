import logging
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger('preprocessor_logger')
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Preprocessor:
    def __init__(self, df: pd.DataFrame, target_col: str = 'target'):
        """
        Inicializa el preprocesador con el DataFrame original.

        Args:
            df: DataFrame a procesar
            target_col: Nombre de la columna target
        """
        self.target_col = target_col
        self.original_df = df.copy()
        self.processed_df = None
        self.cor_target = None
        self.product_cols = None

        # Atributos para exportar
        self.zones_ = None
        self.zone_mapper_df = None
        self.cols_to_drop_ = []
        self.power_params_ = None
        self.skewed_cols_ = []
        self.high_corr_df_products = None

        logger.info(f"Preprocessor inicializado - Shape original: {self.original_df.shape}")

    def _log_shape_change(self, previous_shape: tuple, operation: str):
        """Registra cambios en el shape del DataFrame."""
        current_shape = self.processed_df.shape
        logger.info(
            f"{operation} - Shape cambiado de {previous_shape} a {current_shape} "
            f"(Filas: {previous_shape[0]} → {current_shape[0]}, "
            f"Columnas: {previous_shape[1]} → {current_shape[1]})"
        )

    def group_sociodemographic_cols(self, sociodemographic_cols: list):
        """
        Agrupa columnas sociodemográficas en una sola columna 'zone'.

        Args:
            sociodemographic_cols: Lista de columnas sociodemográficas a agrupar
        """
        logger.info(f"Iniciando agrupación de columnas sociodemográficas: {sociodemographic_cols}")

        previous_shape = self.original_df.shape

        # Crear columna zone basada en agrupación
        self.original_df['zone'] = self.original_df.groupby(sociodemographic_cols).ngroup() + 1

        # Crear mapper de zonas
        expanded_cols = sociodemographic_cols + ['zone']
        self.zone_mapper_df = self.original_df[expanded_cols].drop_duplicates()

        self.zones_ = [str(col) for col in self.zone_mapper_df['zone'].unique()]

        # Eliminar columnas originales
        self.original_df.drop(columns=sociodemographic_cols, inplace=True)
        self.processed_df = self.original_df.copy()

        logger.info(f"Agrupación completada - {len(self.zone_mapper_df)} zonas únicas creadas")
        self._log_shape_change(previous_shape, "Agrupación sociodemográfica")

    def remove_duplicates(self):
        """Elimina duplicados exactos del DataFrame."""
        logger.info("Iniciando eliminación de duplicados exactos")

        previous_shape = self.processed_df.shape
        initial_rows = len(self.processed_df)

        self.processed_df = self.processed_df.drop_duplicates()
        final_rows = len(self.processed_df)
        removed_rows = initial_rows - final_rows

        logger.info(f"Duplicados eliminados: {removed_rows} filas removidas")
        self._log_shape_change(previous_shape, "Eliminación de duplicados exactos")

    def handle_complex_duplicates(self):
        """
        Maneja duplicados complejos donde las filas son idénticas excepto por el target.
        Conserva la moda del target en casos de múltiples duplicados.
        """
        logger.info("Iniciando manejo de duplicados complejos")

        df = self.processed_df.copy()
        previous_shape = df.shape
        initial_rows = len(df)

        # Agrupar filas duplicadas excepto por target
        columns = list(set(df.columns) - {self.target_col})
        grupos_duplicados = df.groupby(columns).groups

        indices_a_eliminar = []
        grupos_procesados = 0

        for fila, indices in grupos_duplicados.items():
            if len(indices) > 1:  # Solo grupos con duplicados
                grupos_procesados += 1
                grupo_actual = df.loc[indices]

                if len(indices) == 2:
                    # Eliminar todo el grupo (ambas filas)
                    indices_a_eliminar.extend(indices)
                else:
                    # Grupos con más de 2 filas: conservar solo la moda
                    moda_target = grupo_actual[self.target_col].mode()

                    if len(moda_target) > 0:
                        moda = moda_target[0]
                        # Conservar solo las filas con target = moda
                        filas_a_eliminar = grupo_actual[grupo_actual[self.target_col] != moda].index
                        indices_a_eliminar.extend(filas_a_eliminar)
                    else:
                        # Si no hay moda clara, eliminar todo el grupo
                        indices_a_eliminar.extend(indices)

        # Crear nuevo DataFrame sin los duplicados problemáticos
        self.processed_df = df.drop(indices_a_eliminar)
        final_rows = len(self.processed_df)
        removed_rows = initial_rows - final_rows

        logger.info(
            f"Manejo de duplicados complejos completado - "
            f"{grupos_procesados} grupos procesados, {removed_rows} filas eliminadas"
        )
        self._log_shape_change(previous_shape, "Manejo de duplicados complejos")

    def get_correlations(self):
        """Calcula correlaciones punto-biserial entre features y target."""
        logger.info("Calculando correlaciones con el target")

        if self.target_col not in self.processed_df.columns:
            raise ValueError(f"Target column '{self.target_col}' no encontrada en el DataFrame")

        cor_target = {}
        features = [col for col in self.processed_df.columns if col != self.target_col]

        for col in features:
            corr, _ = pointbiserialr(self.processed_df[col], self.processed_df[self.target_col])
            cor_target[col] = corr

        self.cor_target = pd.Series(cor_target).sort_values(key=abs, ascending=False)

        logger.info(
            f"Correlaciones calculadas - "
            f"Rango: [{self.cor_target.min():.3f}, {self.cor_target.max():.3f}], "
            f"Top 3: {self.cor_target.head(3).to_dict()}"
        )

    def get_high_pair_correlations(self, product_cols: list):
        """
        Identifica pares de variables con alta correlación entre sí.

        Args:
            product_cols: Lista de columnas de productos a analizar
        """
        logger.info(f"Buscando correlaciones altas entre {len(product_cols)} columnas de productos")

        # Verificar que las columnas existen
        missing_cols = set(product_cols) - set(self.processed_df.columns)
        if missing_cols:
            raise ValueError(f"Columnas no encontradas: {missing_cols}")

        corr_abs_products = self.processed_df[product_cols].corr().abs()

        high_corr_pairs_products = []
        for i in range(len(corr_abs_products.columns)):
            for j in range(i + 1, len(corr_abs_products.columns)):
                correlation = corr_abs_products.iloc[i, j]
                if correlation > 0.7:
                    high_corr_pairs_products.append({
                        'var1': corr_abs_products.columns[i],
                        'var2': corr_abs_products.columns[j],
                        'correlation': correlation
                    })

        self.high_corr_df_products = pd.DataFrame(high_corr_pairs_products).sort_values(
            'correlation', ascending=False
        )

        # Filtrar correlaciones muy altas
        high_corr_count = len(self.high_corr_df_products[self.high_corr_df_products['correlation'] > 0.95])
        self.product_cols = product_cols

        logger.info(
            f"Análisis de correlación completado - "
            f"{len(high_corr_pairs_products)} pares con correlación > 0.7, "
            f"{high_corr_count} pares con correlación > 0.95"
        )

    def drop_high_correlated_cols(self):
        """Elimina columnas altamente correlacionadas, conservando las más relevantes."""
        logger.info("Iniciando eliminación de columnas altamente correlacionadas")

        if self.high_corr_df_products is None:
            raise ValueError("Debe ejecutar get_high_pair_correlations primero")

        previous_shape = self.processed_df.shape

        high_corr_filtered = self.high_corr_df_products[self.high_corr_df_products['correlation'] > 0.95]

        for _, row in high_corr_filtered.iterrows():
            var1 = row['var1']
            var2 = row['var2']

            # Comparar la correlación absoluta con el target
            corr_var1 = abs(self.cor_target[var1])
            corr_var2 = abs(self.cor_target[var2])

            # Quedarse con la columna más correlacionada, eliminar la otra
            if corr_var1 < corr_var2:
                col_to_drop = var1
                col_to_keep = var2
            else:
                col_to_drop = var2
                col_to_keep = var1

            if col_to_drop not in self.cols_to_drop_:
                self.cols_to_drop_.append(col_to_drop)
                logger.debug(f"Marcada para eliminar: {col_to_drop} (corr: {corr_var1:.3f}) "
                             f"vs {col_to_keep} (corr: {corr_var2:.3f})")

        self.cols_to_drop_ = list(set(self.cols_to_drop_))
        self.product_cols = list(set(self.product_cols) - set(self.cols_to_drop_))

        # Eliminar columnas
        columns_before_drop = set(self.processed_df.columns)
        self.processed_df = self.processed_df.drop(columns=self.cols_to_drop_)
        columns_after_drop = set(self.processed_df.columns)
        dropped_columns = columns_before_drop - columns_after_drop

        final_cols = len(self.processed_df.columns)

        logger.info(
            f"Eliminación de columnas correlacionadas completada - "
            f"{len(dropped_columns)} columnas eliminadas: {list(dropped_columns)}"
        )
        self._log_shape_change(previous_shape, "Eliminación de columnas correlacionadas")

    def correct_skewness(self):
        """
        Corrige asimetría en las columnas usando transformación Yeo-Johnson.

        Args:
            product_cols: Lista de columnas de productos a transformar
        """
        logger.info("Iniciando corrección de asimetría")

        # Calcular asimetría inicial
        skewness_before = self.processed_df[self.product_cols].skew()
        self.skewed_cols_ = skewness_before[abs(skewness_before) > 0.5].index.tolist()

        if not self.skewed_cols_:
            logger.info("No se encontraron columnas con asimetría significativa (> 0.5)")
            return

        logger.info(f"{len(self.skewed_cols_)} columnas con asimetría > 0.5: {self.skewed_cols_}")

        # Aplicar transformación Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        self.processed_df[self.skewed_cols_] = pt.fit_transform(self.processed_df[self.skewed_cols_])

        # Calcular asimetría después de la transformación
        skewness_after = self.processed_df[self.skewed_cols_].skew()

        # Guardar parámetros
        if hasattr(pt, "lambdas_"):
            self.power_params_ = dict(zip(self.skewed_cols_, pt.lambdas_))
        else:
            self.power_params_ = {}

        logger.info(
            f"Corrección de asimetría completada - "
            f"Asimetría promedio: {skewness_before.mean():.3f} → {skewness_after.mean():.3f}"
        )

    def apply_one_hot(self):
        """Aplica one-hot encoding a la columna 'zone'."""
        logger.info("Aplicando one-hot encoding a la columna 'zone'")

        if 'zone' not in self.processed_df.columns:
            raise ValueError("Columna 'zone' no encontrada para one-hot encoding")

        previous_shape = self.processed_df.shape

        # Convertir a entero y aplicar one-hot
        self.processed_df['zone'] = self.processed_df['zone'].astype(int)
        zone_dummies = pd.get_dummies(self.processed_df['zone'], prefix='zone')

        # Concatenar y eliminar columna original
        self.processed_df = pd.concat([
            self.processed_df.drop('zone', axis=1),
            zone_dummies
        ], axis=1)

        logger.info(f"One-hot encoding completado - {len(zone_dummies.columns)} columnas zone creadas")
        self._log_shape_change(previous_shape, "One-hot encoding")

    def apply_preprocess(self, sociodemographic_cols: list, product_cols: list) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de preprocesamiento.

        Args:
            sociodemographic_cols: Columnas sociodemográficas a agrupar
            product_cols: Columnas de productos para análisis de correlación

        Returns:
            DataFrame procesado
        """
        logger.info("=== INICIANDO PIPELINE COMPLETO DE PREPROCESAMIENTO ===")
        logger.info(f"Columnas sociodemográficas: {sociodemographic_cols}")
        logger.info(f"Columnas de productos: {product_cols}")

        # Pipeline de procesamiento
        self.group_sociodemographic_cols(sociodemographic_cols)
        self.remove_duplicates()
        self.handle_complex_duplicates()
        self.get_correlations()
        self.get_high_pair_correlations(product_cols)
        self.drop_high_correlated_cols()
        self.correct_skewness()
        self.apply_one_hot()

        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        logger.info(f"Shape final del DataFrame: {self.processed_df.shape}")
        logger.info(f"Columnas eliminadas: {len(self.cols_to_drop_)}")
        logger.info(f"Columnas transformadas: {len(self.skewed_cols_)}")

        return self.processed_df

    def get_preprocessing_summary(self) -> dict:
        """Retorna un resumen del proceso de preprocesamiento."""
        return {
            'original_shape': self.original_df.shape,
            'processed_shape': self.processed_df.shape if self.processed_df is not None else None,
            'zones_created': len(self.zone_mapper_df) if self.zone_mapper_df is not None else 0,
            'columns_dropped': len(self.cols_to_drop_),
            'columns_skewness_corrected': len(self.skewed_cols_),
            'high_correlation_pairs': len(self.high_corr_df_products) if self.high_corr_df_products is not None else 0
        }
