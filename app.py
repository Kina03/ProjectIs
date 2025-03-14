import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler



url1 = "https://raw.githubusercontent.com/Kina03/ProjectIs/refs/heads/main/Global%20Peace%20Index%202023.csv"
data1 = pd.read_csv(url1)
url2 = "https://raw.githubusercontent.com/Kina03/ProjectIs/refs/heads/main/South_Asian_dataset.csv"
data2 = pd.read_csv(url2)


def page1():
    st.title("Machine Learning")
    st.header("About Datasets 📈📑")
    st.write("📄Dataset 1: Global Peace Index 2023📑")
    st.write("ชุดข้อมูล 'Global Peace Index 2023' ประกอบด้วยข้อมูลที่รวบรวมจากสถาบันเศรษฐศาสตร์และสันติภาพ (IEP) ที่เกี่ยวข้องกับแนวโน้มของสันติภาพ ")
    
    st.text("Example features of Global Peace Index 2023 in dataset ")
    st.write(data1.head())

    st.markdown("""
    🔍 Source : [Global Peace Index 2023](https://www.kaggle.com/datasets/ddosad/global-peace-index-2023/data)
    ### Features of dataset :
    - Country : ชื่อประเทศ
    - iso3c : รหัสประเทศแบบ 3 ตัวอักษร
    - Overall Scores : คะแนนรวม
    - Safety and Security : ระดับความปลอดภัยและความมั่นคง
    - Ongoing Conflict : ความขัดแย้งที่ยังคงดำเนินอยู่
    - Militarian : ระดับของการทหาร
    """)       

    st.header("👩🏻‍💻 Data preparation :")
    st.markdown("""
    - อ่านข้อมูลจากไฟล์ CSV 
    - ตรวจสอบค่าที่หายไป (NaN) โดยใช้ df.isnull().sum() และทำการลบแถวที่มีค่าเป็น NaN ใน Overall Scores ออกเพื่อให้ข้อมูลสมบูรณ์พร้อมสำหรับการวิเคราะห์
    - เลือกเฉพาะคอลัมน์ที่ใช้งาน โดย dataset แรก จะใช้แค่คอลัมน์ 'Overall Scores', 'Safety and Security', 'Ongoing Conflict'
    - สร้าง Label (Target Variable)ชื่อว่า Score Category สำหรับ Classification โดยจะแบ่ง Overall Scores ออกเป็น 3 กลุ่ม ได้แก่ 'Low'(ความสงบต่ำ), 'Medium'(ความสงบปานกลาง), 'High'(ความสงบสูง)
    - แปลง Label เป็นตัวเลขโดยใช้ LabelEncoder() เปลี่ยนค่า 'Low', 'Medium', 'High' ให้เป็นตัวเลข 0, 1, 2
    - ทำการแยกระหว่าง Features(X) ▶️ 'Safety and Security', 'Ongoing Conflict' และ Labels(y) ▶️ 'Score Category'(0, 1, 2) เพื่อใช้ในการแบ่งข้อมูลเป็นชุด Train/Test โดยจะแบ่ง Train = 80% และ Test = 20%
    - ใช้ Standardization เพื่อปรับข้อมูลให้เป็นมาตรฐาน โดยทำให้ข้อมูลมีค่าเฉลี่ยเท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐานเท่ากับ 1

""")

    st.header("🛠️ การพัฒนา Model Machine Learning")
    st.markdown("""
    - **KNeighbors Classifier :**
        * KNN เป็นโมเดล Machine Learning ประเภท Supervised แบบ Classification จำเป็นต้องมี Data set ที่มีเฉลย(Label) ด้วย โดยอาศัยหลักการของระยะห่างระหว่างข้อมูลในการจำแนกประเภท โดยเราจะใช้ข้อมูล Global Peace Index 2023 เพื่อจำแนกระดับความสงบสุขของประเทศต่าง ๆ ออกเป็น 3 กลุ่ม มีการเตรียมข้อมูล แปลงค่าข้อมูล 
                และใช้ PCA เพื่อลดมิติของข้อมูล ก่อนที่จะนำไปฝึกและทดสอบโมเดล ซึ่งช่วยให้สามารถจำแนกประเทศตามระดับความสงบสุขได้อย่างมีประสิทธิภาพ และประเมินประสิทธิภาพผ่าน Accuracy Score และ Confusion Matrix มีการแสดงผลลัพธ์เป็นกราฟ Scatter plot เพื่อให้เห็นว่าข้อมูลถูกจัดกลุ่มอย่างไร และจุดผิดพลาดของโมเดลอยู่ที่ไหน 
        * ขั้นตอนการพัฒนา
            * ทำการโหลดและเตรียมข้อมูล
            * โมเดล KNeighbors Classifier สร้างขึ้นโดยใช้ KNeighborsClassifier(n_neighbors=5) ใช้ค่าเริ่มต้นของพารามิเตอร์ k เท่ากับ 5 โดยโมเดลจะทำการฝึกกับข้อมูล X_train และ y_train โดยอาศัยระยะห่างระหว่างข้อมูลเพื่อกำหนดกลุ่มที่เหมาะสมที่สุดเพื่อใช้ในการแยกประเภทข้อมูล
            * ใช้ train_test_split แบ่งข้อมูลเป็น 80% สำหรับฝึก (Training Set) และ 20% สำหรับทดสอบ (Testing Set) โดยตั้งค่า stratify=y เพื่อให้มั่นใจว่าการกระจายของกลุ่มยังคงสมดุลกันในชุด Train และ Test
            * เมื่อโมเดลได้รับการฝึกแล้ว จะใช้ชุดข้อมูลทดสอบ X_test ในการทำนาย โดยโมเดลจะทำนายว่าประเทศไหนควรอยู่ในกลุ่ม Low, Medium หรือ High ตามลักษณะของตัวแปร 'Safety and Security' และ 'Ongoing Conflict'
            * เมื่อได้ค่าที่ทำนายแล้ว จะมีการใช้ Accuracy Score และ Confusion Matrix เพื่อตรวจสอบความถูกต้องของโมเดล
            * มีการแสดงผลลัพธ์เป็นกราฟ Scatter plot เพื่อให้เห็นว่าข้อมูลถูกจัดกลุ่มอย่างไร และจุดผิดพลาดของโมเดลอยู่ที่ไหน 
                

    - **SVM Classifier :**
        * SVM หรือ Support Vector Machine เป็นโมเดล Machine Learning ที่ใช้ในการจำแนกข้อมูล(Classification) หรือแบ่งกลุ่มข้อมูลโดยจะสร้างเส้นที่ใช้แบ่งกลุ่มข้อมูล (Hyperplane) และหาเส้นทางที่ดีที่สุด 
                โมเดล SVM ได้รับการพัฒนาโดยใช้ของ KNeighbors Classifier และมีการประเมินประสิทธิภาพผ่าน Accuracy Score และ Confusion Matrix
        * ขั้นตอนการพัฒนา
            * ทำการโหลดและเตรียมข้อมูล
            * สร้างโมเดล SVM โดยใช้ SVC ซึ่งเป็นคลาสที่ใช้สำหรับจำแนกประเภท และตั้งค่าพารามิเตอร์ random_state=42 เพื่อให้ผลลัพธ์ของโมเดลทำซ้ำได้ และใช้ค่าเริ่มต้นของพารามิเตอร์ (Kernel = 'rbf')
            * ใช้ train_test_split เพื่อแบ่งข้อมูลออกเป็น 2 ส่วน คือ 
                * 1. X_train และ X_test ▶️ เป็นชุดข้อมูลที่ใช้ในการฝึกและทกสอบโมเดล โดยใช้ตัวแปร 'Safety and Security' และ 'Ongoing Conflict' 
                * 2. y_train และ y_test ▶️ คือผลลัพธ์ที่ใช้เปรียบเทียบในการจำแนกประเภทของโมเดล มีการใช้ stratify=y เพื่อช่วยให้ข้อมูลแต่ละกลุ่มมีสัดส่วนที่เท่ากัน
            * ก่อนที่จะนำข้อมูลไปใช้กับโมเดล จะต้องมีการทำ Standardization โดยใช้ StandardScaler เพื่อทำให้ค่าข้อมูลอยู่ในช่วงเหมาะสม มีการใช้ fit_transform กับ X_train และ transform กับ X_test เพื่อให้มั่นใจว่าข้อมูลทั้งสองชุดใช้มาตรฐานเดียวกัน
            * เมื่อได้ค่าที่ทำนายแล้ว จะมีการใช้ Accuracy Score และ Confusion Matrix เพื่อตรวจสอบความถูกต้องของโมเดล มีการแสดงผลลัพธ์เป็นกราฟ Scatter plot เพื่อให้เห็นว่าข้อมูลถูกจัดกลุ่มอย่างไร และจุดผิดพลาดของโมเดลอยู่ที่ไหน 
            * มีการแสดงผลลัพธ์เป็นกราฟ Scatter plot เพื่อให้เห็นว่าข้อมูลถูกจัดกลุ่มอย่างไร และจุดผิดพลาดของโมเดลอยู่ที่ไหน 
                

    - **KMeans :**
        * KMeans เป็น Unsupervised Learning คือการเรียนรู้แบบไม่ต้องสอน ไม่มีคำตอบตายตัว โดยหน้าที่หลักของ K-means คือการแบ่งกลุ่ม แบบ Clustering ซึ่งการแบ่งกลุ่มในลักษณะนี้จะใช้พื้นฐานทางสถิติ ซึ่งหน้าที่ของ clustering คือการจับกลุ่มของข้อมูลที่มีลักษณะใกล้เคียงกันเป็นกลุ่มเดียวกัน
        * ขั้นตอนการพัฒนา
            * ทำการโหลดและเตรียมข้อมูล
            * ทำการหาจำนวน Cluster ที่เหมาะสมโดยใช้ Elbow Method ซึ่งเป็นเทคนิคที่ช่วยกำหนด Cluster 
                โดยคำนวณค่า WCSS สำหรับจำนวน Cluster ที่ต่างกัน และแสดงกราฟเพื่อตรวจสอบตำแหน่งที่มีการลดลงอย่างชัดเจน ซึ่งจะแสดงจำนวน Cluster ที่เหมาะสม
            * เมื่อเลือกจำนวน Cluster แล้วใช้ KMeans(n_clusters=3, random_state=42, n_init=10) เพื่อทำการจัดกลุ่มประเทศออกเป็น 3 กลุ่มตามค่าคะแนนที่ได้
            *  มีการกำหมดชื่อกลุ่ม Cluster เพื่อให้เข้าใจง่ายขึ้นโมเดลจะถูกจัดลำดับตามค่าเฉลี่ยของ "Overall Scores" และกำหนดชื่อกลุ่ม ได้แก่ :
                * Highly Peaceful: ประเทศที่มีระดับความสงบสุขสูงสุด
                * Moderately Peaceful: ประเทศที่มีระดับความสงบสุขปานกลาง
                * High Conflict: ประเทศที่มีระดับความขัดแย้งสูง
            * มีการนับจำนวนประเทศที่อยู่ในแต่ละ Cluster โดยใช้ value_counts() เพื่อดูการกระจายตัวของแต่ละ Cluster ว่ามีความสมดุลมั้ย
            * พล็อตกราฟแสดงความสัมพันธ์ระหว่าง "Overall Scores" และ "Safety and Security" พร้อมกับการระบายสีตามคลัสเตอร์ที่โมเดลได้จัดกลุ่มไว้ ซึ่งช่วยให้เห็นภาพรวมของกลุ่มที่เกิดขึ้นจากการ Clustering
""")
        
                


def page2():

    st.title("Neural Network")
    st.header("About Datasets 📈📑")
    st.write("📄Dataset 2: South Asian Growth & Development Data (2000-23)📑")
    st.write("ชุดข้อมูล 'South Asian Growth & Development Data (2000-23)' ประกอบด้วยข้อมูล GDP อัตราการว่างงาน อัตราการรู้หนังสือ การใช้พลังงาน ตัวชี้วัดการกำกับดูแล และอื่นๆ ของประเทศในเอเชียใต้ตั้งแต่ปี 2543 ถึง 2566 ช่วยให้สามารถวิเคราะห์การเติบโตของแต่ละประเทศได้")
    
    st.text("Example features of South Asian Growth & Development Data in dataset ")
    st.write(data2.head())    
    st.markdown("""
    🔍 Source : [South Asian Growth & Development Data (2000-23)](https://www.kaggle.com/datasets/rezwananik/south-asia-growth-and-development-data-2000-23)
    ### Features of dataset :
    - **Country :** ชื่อประเทศ
    - **Year :** ปี
    - **GDP (current US$) :** GDP (ดอลลาร์สหรัฐในปัจจุบัน)
    - **GDP growth (annual %) :** อัตราการเติบโตของ GDP (ร้อยละต่อปี)
    - **GDP per capita (current US$) :** GDP ต่อหัว (ดอลลาร์สหรัฐในปัจจุบัน)
    - **etc.**
    """)      


    st.header("👩🏻‍💻 Data preparation :")
    st.markdown("""
    - อ่านข้อมูลจากไฟล์ CSV 
    - เลือกเฉพาะคอลัมน์ที่ใช้งาน โดย dataset ที่สอง จะใช้แค่คอลัมน์ 'Country','Year', 'GDP (current US$)'
    - แปลงค่าที่เป็น ".." ให้เป็น NaN
    - แปลงคอลัมน์ 'GDP (current US)' ให้เป็นตัวเลขด้วย pd.to_numeric(df_selected["GDP (current US$)"], errors="coerce")
    - ตรวจสอบค่าที่หายไป (NaN) โดยใช้ df.isnull().sum() และทำการลบแถวที่มีค่าเป็น NaN ใน GDP (current US$) ออกเพื่อให้ข้อมูลสมบูรณ์พร้อมสำหรับการวิเคราะห์ ลบข้อมูลที่ซ้ำกันด้วย drop_duplicates() เพื่อให้แน่ใจว่าไม่มีข้อมูลที่ผิดพลาด ใช้ Interpolation (df.interpolate()) เพื่อเติมค่าที่หายไปแบบเส้นตรง
            กรองค่าผิดปกติ (Outliers) ด้วย IQR (Interquartile Range) เพื่อลบค่าที่แตกต่างจากค่าปกติมากเกินไป
    - ใช้ MinMaxScaler เพื่อปรับค่า GDP (current US$) ให้อยู่ในช่วง 0-1 เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น
    - เตรียมข้อมูลสำหรับทำ LSTM โดยจะใช้ข้อมูลย้อนหลัง 3 ปีที่ผ่านมา เพื่อมาทำนาย GDP ในปีถัด ๆ ไป

""")
    
    st.header("🛠️ การพัฒนา Model Neural Network")
    st.markdown("""
    - **LSTM Neural Network :**
        * LSTM  เป็นประเภทหนึ่งของ Recurrent Neural Network (RNN) ถูกออกแบบให้จดจำ Patterns ในช่วงเวลานาน ๆ มีประสิทธิภาพในการทำนายที่เป็นแบบ Sequential  เนื่องจากสามารถเก็บข้อมูลก่อนหน้าและนำมาใช้ประมวลผลได้
        * ขั้นตอนการพัฒนา
            * ทำการโหลดและเตรียมข้อมูล
            * สร้างชุดข้อมูลที่เป็นลำดับเวลา (Sequence Data) โดยจะใช้ข้อมูลย้อนหลัง 3 ปี(SEQ_LENGTH = 3) เพื่อเป็นอินพุต สำหรับโมเดล แบ่งข้อมูลออกเป็น
                * 1. X ▶️ ข้อมูลย้อนหลัง 3 ปี
                * 2. y ▶️ GDP ของปีถัดไปที่ต้องการทำนาย
            * ใช้ Interquartile Range (IQR) Method เพื่อตรวจจับและลบค่าผิดปกติ (outliers) ออกจากข้อมูล GDP ของประเทศ
            * แบ่งข้อมูลเป็น 80% สำหรับ Train และ 20% สำหรับ Test ใช้ Train set ในการฝึกโมเดล และใช้ Test set เพื่อตรวจสอบประสิทธิภาพของโมเดล
            * ใช้โครงข่ายประสาทเทียมแบบ LSTM (Long Short-Term Memory)  โดย
                * LSTM Layer 1 มี 50 Neurons และใช้ relu เป็น activation function
                * LSTM Layer 2 มี 50 Neurons และใช้ relu เป็น activation function
                * Dense Layer (Fully Connected Layer): ใช้เป็น output layer ที่มี 1 Neuron
                * ใช้ Adam Optimizer และ Mean Squared Error (MSE) เป็น loss function
            * กำหนดให้มี 100 epochs และใช้ batch size = 8 ใช้ชุดข้อมูลฝึกสอน (X_train, y_train) และตรวจสอบผลลัพธ์ด้วยชุดข้อมูลทดสอบ (X_test, y_test)
            * การทำนาย GDP ของอนาคต
                * ใช้ค่า GDP ล่าสุดเป็นจุดเริ่มต้น
                * คำนวณ GDP ของปีถัดไปโดยใช้โมเดล LSTM 
""")
    

def page3():

    st.title("Machine Learning Demo")
    st.header("🌍 Global Peace Index Classification and Clustering")

    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/Kina03/ProjectIs/refs/heads/main/Global%20Peace%20Index%202023.csv"
        df = pd.read_csv(url)
        df['Overall Scores'].fillna(df['Overall Scores'].mean(), inplace=True)
        return df
    df = load_data()

    #เลือกปีที่ต้องการ
    year_list = sorted(df["year"].unique(), reverse=True)
    selected_year = st.selectbox("เลือกปีที่ต้องการดูข้อมูล:", year_list)

    df_selected = df[df["year"] == selected_year].copy().reset_index(drop=True)

    # สร้าง Label Encoding สำหรับ Score Category
    df_selected['Score Category'] = pd.qcut(df_selected['Overall Scores'], q=3, labels=['Low', 'Medium', 'High'])
    df_selected['Score Category'] = LabelEncoder().fit_transform(df_selected['Score Category'])

    # เตรียมข้อมูลสำหรับ Classification
    X = df_selected[['Safety and Security', 'Ongoing Conflict']]
    y = df_selected['Score Category']

    if len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

        # Train และ แสดงผล
        def train_and_plot(model, name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.subheader(f"{name} Model Performance for Year {selected_year}")
            st.write(f"✅ **Accuracy:** {accuracy:.2f}")

            # แสดง Confusion Matrix
            st.write(f"📊 **Confusion Matrix ({name}):**")
            st.write(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                index=['Actual Low', 'Actual Medium', 'Actual High'],
                                columns=['Pred Low', 'Pred Medium', 'Pred High']))

            # สร้าง Scatter Plot
            cmap = ListedColormap(['Tomato', 'PaleGreen', 'Gold'])
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, alpha=0.6, s=90)

            # จุดที่ทำนายผิด
            incorrect_indices = np.where(y_pred != y_test)[0]
            ax.scatter(X_test[incorrect_indices, 0], X_test[incorrect_indices, 1], 
                    marker='x', color='RoyalBlue', label='Misclassified', s=100)

            #ชื่อประเทศ
            country_test = df_selected.iloc[X_test.shape[0]:]["Country"].values
            num_points = min(len(X_test), len(country_test))
            for i in range(num_points):
                ax.annotate(country_test[i], (X_test[i, 0], X_test[i, 1]), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=6)

            plt.title(f'{name} Predictions for Year {selected_year} with Misclassified Points')
            plt.xlabel('Safety and Security')
            plt.ylabel('Ongoing Conflict')

            legend_elements = scatter.legend_elements()
            plt.legend(handles=legend_elements[0], labels=['Low', 'Medium', 'High'], loc="upper left", title="Peace Level")
            st.pyplot(fig)

        # Train และแสดงผล 
        train_and_plot(KNeighborsClassifier(n_neighbors=5), "KNN")
        train_and_plot(SVC(kernel='rbf', random_state=42), "SVM")

    # เตรียมข้อมูลสำหรับ Clustering
    features = ["Overall Scores", "Safety and Security", "Ongoing Conflict", "Militarian"]
    df_clean = df.dropna(subset=["Overall Scores"]).copy()  # ลบค่า NaN

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_clean[features])

    #ใช้ Elbow Method หา K ที่เหมาะสม
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_cluster_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, wcss, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method to Determine Optimal K')
    st.pyplot(fig)

    # ใช้ K-Means แบ่งเป็น 3 กลุ่ม
    features = ["Overall Scores", "Safety and Security", "Ongoing Conflict", "Militarian"]
    df_cluster = df[df["year"] == selected_year].dropna(subset=["Overall Scores"]).copy()

    scaler = StandardScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster[features])

    st.subheader(f"KMeans Model Performance for Year {selected_year}")
    features = ["Overall Scores", "Safety and Security", "Ongoing Conflict", "Militarian"]
    df_cluster_scaled = StandardScaler().fit_transform(df_selected[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_selected["Cluster"] = kmeans.fit_predict(df_cluster_scaled)

    cluster_order = df_selected.groupby("Cluster")["Overall Scores"].mean().sort_values().index.tolist()
    cluster_labels = {
        cluster_order[0]: "Highly Peaceful",  #สงบมาก
        cluster_order[1]: "Moderately Peaceful",
        cluster_order[2]: "High Conflict"  #ความขัดแย้งมาก
    }

    df_selected["Cluster Label"] = df_selected["Cluster"].map(cluster_labels)

    cluster_colors = {
        "Highly Peaceful": 'PaleGreen',
        "Moderately Peaceful": 'Gold',
        "High Conflict": 'Tomato'}

    # เลือกประเทศ
    selected_country = st.selectbox("🌍 เลือกประเทศ : ", df_selected["Country"].sort_values().unique())

    # ดึงข้อมูลประเทศที่เลือก
    country_data = df_selected[df_selected["Country"] == selected_country]

    if not country_data.empty:
        cluster_label = country_data["Cluster Label"].values[0]
    
    if cluster_label == "Highly Peaceful":
        st.success(f"🌱 **{selected_country} อยู่ในกลุ่ม:** {cluster_label}")
    elif cluster_label == "Moderately Peaceful":
        st.warning(f"🌞 **{selected_country} อยู่ในกลุ่ม:** {cluster_label}")
    else:
        st.error(f"⚔️ **{selected_country} อยู่ในกลุ่ม:** {cluster_label}") 

    #แสดงกราฟ Scatter Plot
    cmap = ListedColormap([cluster_colors[cluster_labels[cluster_order[0]]],  # Highly Peaceful (เขียว)
                            cluster_colors[cluster_labels[cluster_order[1]]],  # Moderately Peaceful (เหลือง)
                            cluster_colors[cluster_labels[cluster_order[2]]]])  # High Conflict (แดง)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_selected["Overall Scores"],
                        df_selected["Safety and Security"],
                        c=df_selected["Cluster"].map(lambda x: cluster_order.index(x)),
                        cmap=cmap, alpha=0.6, s=90)

    ax.set_title("Clusters based on Overall Scores and Safety & Security")
    ax.set_xlabel("Overall Scores")
    ax.set_ylabel("Safety and Security")

    # แสดงประเทศที่เลือกในกราฟ
    if not country_data.empty:
        country_x = country_data["Overall Scores"].values[0]
        country_y = country_data["Safety and Security"].values[0]
        ax.scatter(country_x, country_y, color='SlateBlue', s=110, edgecolor='DarkSlateBlue', marker='o', label=selected_country)
        ax.annotate(selected_country, (country_x, country_y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')

    plt.legend(handles=scatter.legend_elements()[0], labels=["Highly Peaceful", "Moderately Peaceful", "High Conflict"], title="Cluster")
    st.pyplot(fig)

    # แสดงข้อมูล
    st.write("📌 **ข้อมูลประเทศที่เลือก:**")
    st.dataframe(country_data[["Country", "Overall Scores", "Safety and Security", "Ongoing Conflict", "Cluster Label"]], use_container_width=True)

    st.write("📊 **K-Means Clustering Results**")
    st.dataframe(df_selected[["Country", "Cluster Label"]].sort_values("Cluster Label"), use_container_width=True)


def page4():

    st.title("Neural Network Demo")
    st.header("📉 South Asian Growth & Development Data (2000-23)")
    url2 = "https://raw.githubusercontent.com/Kina03/ProjectIs/refs/heads/main/South_Asian_dataset.csv"
    data2 = pd.read_csv(url2)

    #เลือกประเทศ
    st.subheader(f"LSTM Neural Network")
    selected_country = st.selectbox("🌍 เลือกประเทศที่ต้องการดู GDP :", data2["Country"].unique())

    # ดึงข้อมูลของประเทศที่เลือก
    df_country = data2[data2["Country"] == selected_country][["Year", "GDP (current US$)"]].dropna()
    df_country["GDP (current US$)"] = df_country["GDP (current US$)"].interpolate(method="linear")
    df_country = df_country.drop_duplicates(subset=["Year"], keep="first")

    # ลบค่าผิดปกติ
    Q1 = df_country["GDP (current US$)"].quantile(0.25)
    Q3 = df_country["GDP (current US$)"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_country = df_country[(df_country["GDP (current US$)"] >= lower_bound) & (df_country["GDP (current US$)"] <= upper_bound)]

    #Scaling ข้อมูล
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_country["GDP_scaled"] = scaler.fit_transform(df_country[["GDP (current US$)"]])

    #เตรียมข้อมูลสำหรับ LSTM
    SEQ_LENGTH = 3
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    data = df_country["GDP_scaled"].values
    X, y = create_sequences(data, SEQ_LENGTH)

    #แบ่ง Train/Test Set
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    #สร้างโมเดล LSTM
    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        LSTM(50, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # เทรนโมเดล
    X_train = X_train.reshape(-1, SEQ_LENGTH, 1)
    X_test = X_test.reshape(-1, SEQ_LENGTH, 1)
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=0)

    # ทำนาย จนถึงปี 2030
    future_years = list(range(df_country["Year"].max() + 1, 2031))
    future_gdp_scaled = []
    last_sequence = X[-1].reshape(1, SEQ_LENGTH, 1)

    for year in future_years:
        next_gdp_scaled = model.predict(last_sequence)[0][0]
        future_gdp_scaled.append(next_gdp_scaled)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_gdp_scaled

    future_gdp_actual = scaler.inverse_transform(np.array(future_gdp_scaled).reshape(-1, 1)).flatten()

    st.subheader(f"🔮 การทำนาย GDP ของ {selected_country} จนถึงปี 2030")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_country["Year"], df_country["GDP (current US$)"], label="Actual GDP", marker="o")
    ax.plot(future_years, future_gdp_actual, label="Predicted GDP", marker="x", linestyle="dashed", color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (US$)")
    ax.legend()
    ax.set_title(f"GDP Forecast for {selected_country} until 2030")
    st.pyplot(fig)

    future_df = pd.DataFrame({"Year": future_years, "Predicted GDP": future_gdp_actual})
    st.dataframe(future_df, use_container_width=True)
 
pg = st.navigation([
    st.Page(page1, title="Machine Learning", icon="🦾"),
    st.Page(page2, title="Neural Network", icon="🧠"),
    st.Page(page3, title="Machine Learning Demo", icon="🤖"),
    st.Page(page4, title="Neural Network Demo", icon="🧬")
])
pg.run()