// Unity built-in shader source. Copyright (c) 2016 Unity Technologies. MIT license (see license.txt)

Shader "YoyoPbrPrac"
{
	Properties{
		 _Color("Color",Color) = (1,1,1,0)
		 _MainTex("MainTex", 2D) = "white" {}
		 _Normal("Normal", 2D) = "bump" {}
		 _CubeMap("CubeMap", Cube) = "_Skybox" {}
		 _Control("R(metallic)G(AO)B(Rough) 注意需要线性空间", 2D) = "white" {}
		 _SpecColor("SpecColor",Color) = (0.2,0.2,0.2)
		 metalpower("metallic",Range(0,1)) = 1
		 smoothpower("smooth",Range(-1,1)) = 1
		 aopower("ao",Range(0,1)) = 1
	}
		SubShader
		 {
			LOD 300
			Blend SrcAlpha OneMinusSrcAlpha
			ZWrite on
			ZTest on

			Tags {
				"RenderType" = "Opaque"
			}

		  Pass {
		  Blend SrcAlpha OneMinusSrcAlpha

		  Cull Back
		  ZWrite on
		  ZTest on
			  Name "FORWARD"
			  Tags {
				  "LightMode" = "ForwardBase"
			  }
			  CGPROGRAM
			  #pragma vertex vert
			  #pragma fragment frag
			  #define UNITY_PASS_FORWARDBASE
			  #define UNITY_LIGHTMAP_DLDR_ENCODING
			  #include "UnityCG.cginc"
			  #include "AutoLight.cginc"
			  #include "Lighting.cginc"
			  #include "UnityStandardBRDF.cginc"

			  #pragma multi_compile_fwdbase_fullshadows
			   #pragma multi_compile LIGHTMAP_OFF LIGHTMAP_ON
			  #pragma multi_compile_fog
			  #pragma target 3.0
			  uniform sampler2D _MainTex; uniform fixed4 _MainTex_ST;
			  uniform sampler2D _Normal; uniform fixed4 _Normal_ST;
			  uniform samplerCUBE _CubeMap;
			  uniform sampler2D _Control; uniform fixed4 _Control_ST;
			  uniform fixed4 _Color;
			  uniform fixed smoothpower;
			  uniform fixed metalpower;

			  fixed aopower;
			  float3 ACESToneMapping(float3 color)
			  {
				  const float A = 2.51;
				  const float B = 0.03;
				  const float C = 2.43;
				  const float D = 0.59;
				  const float E = 0.14;
				  return (color * (A * color + B)) / (color * (C * color + D) + E);
			  }
			  /////////////////
			  struct VertexInput {
				  float4 vertex : POSITION;
				  float3 normal : NORMAL;
				  float4 tangent : TANGENT;
				  half2 texcoord0 : TEXCOORD0;
			  };
			  struct VertexOutput {
				  float4 pos : SV_POSITION;
				  half2 uv0 : TEXCOORD0;
				  float4 normalDir : TEXCOORD1;
				  float4 tangentDir : TEXCOORD2;
				  float4 bitangentDir : TEXCOORD3;
				  LIGHTING_COORDS(4,5)
				  UNITY_FOG_COORDS(6)
			  };
			  VertexOutput vert(VertexInput v) {
				  VertexOutput o = (VertexOutput)0;
				  o.uv0 =  TRANSFORM_TEX(v.texcoord0, _MainTex);
				  o.normalDir.xyz = UnityObjectToWorldNormal(v.normal);
				  o.tangentDir.xyz = normalize(mul(unity_ObjectToWorld, fixed4(v.tangent.xyz, 0.0)).xyz);
				  o.bitangentDir.xyz = normalize(cross(o.normalDir, o.tangentDir) * v.tangent.w);

				  float3 posWorld = mul(unity_ObjectToWorld, v.vertex);
				  o.normalDir.w = posWorld.x;
				  o.tangentDir.w = posWorld.y;
				  o.bitangentDir.w = posWorld.z;
				  
				  o.pos = UnityObjectToClipPos(v.vertex);
				  
				  UNITY_TRANSFER_FOG(o,o.pos);
				  TRANSFER_VERTEX_TO_FRAGMENT(o)//根据该pass处理的光源类型（ spot 或 point 或 directional ）来计算光源坐标的具体值，以及进行和 shadow 相关的计算等
				  return o;
			  }
			  float4 frag(VertexOutput i) : COLOR {
				  fixed3x3 tangentTransform = fixed3x3(i.tangentDir.xyz, i.bitangentDir.xyz, i.normalDir.xyz);
				  //↑↑↑↑↑↑这里是做按行排列，所以是从 世界坐标系转法线坐标系的矩阵 ↑↑↑↑↑↑

				  fixed3 posWorld = fixed3(i.normalDir.w, i.tangentDir.w, i.bitangentDir.w);
				  fixed3 tangentNormal = UnpackNormal(tex2D(_Normal, i.uv0));
				  fixed3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - posWorld.xyz);
				  float3 worldNormal = normalize(mul(tangentNormal, tangentTransform));//左乘到世界坐标系
				  fixed3 viewReflectDirection = reflect(-viewDirection, worldNormal);//视角的对称方向，用来采CubeMap
				  fixed3 lightDirection = normalize(_WorldSpaceLightPos0.xyz);
				  fixed3 halfDirection = normalize(viewDirection + lightDirection);

				  fixed4 controlMap = tex2D(_Control, i.uv0);//r metallic , g ao , b roughness
				  fixed metalic = controlMap.r *metalpower ;
				  fixed roughness =  controlMap.b * (1-smoothpower);  
				  fixed ao = controlMap.g ;
				  float perceptualRoughness = sqrt(roughness);
				  
				  fixed4 mainColor = tex2D(_MainTex,i.uv0);
				  fixed3 albedo = mainColor.rgb;

				  float LdotN = max(0.001, dot(lightDirection, worldNormal));//避免被除以0
				  float NdotV = max(0.001,dot(worldNormal, viewDirection));//避免被除以0
				  float NdotH = max(0,dot(worldNormal, halfDirection));
				  float LdotH = max(0, dot(lightDirection, halfDirection));

				  //fresnel
				  //unity_ColorSpaceDielectricSpec 是一个很暗的值,模拟金属的黯淡散射
				  float3 F0 = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metalic);
				  float3 fresnel = F0 + (1 - F0)  * pow(1 - LdotH, 5); // Unity 本身使用的是 LdotH 据说是对GGXTerm方法的修正emm
				  //便于理解的话 可以写成这个 
				  //F0 = lerp(unity_ColorSpaceDielectricSpec , 1, metalic) * albedo
				  //fresnel = lerp(F0 , 1 , 1- (nv)^5)，可以直观的看出来，metalic越大 -> F0越大->fresnel 的初始值越 大

				  //另外Unity有下面的这个函数
				  //inline half3 FresnelLerp(half3 F0, half3 F90, half cosA)
				  //{
				  // half t = Pow5(1 - cosA);   // ala Schlick interpoliation
				  // return lerp(F0, F90, t);
				  //} 
				  // 说明我们在计算上面的fresnel的时候，相当于令 F90 = 1
				  //F0 应该可以理解为 我们正直的看一个平面时，菲涅尔参数的值
				  //而 F90 就是我们与平面相平齐的时候，菲涅尔参数的值，这是一个非常大的值，在这里就直接视为1 了


				  //BRDF - diffuse
				  fixed kd = (1 - fresnel) * (1 - metalic);
				  fixed3 diffuse = kd * max(0, LdotN)*_LightColor0.rgb * albedo ;
				  diffuse = lerp(diffuse, diffuse*ao, aopower);
				  
				  //接受影子
				  fixed attenuation = LIGHT_ATTENUATION(i);
				  diffuse *= attenuation;

				  //BRDF - Cook-Torrance
				  //几何衰减项 ,体现光在粗糙表面上反射时的损耗，越粗糙，损耗越大
				  half V = SmithBeckmannVisibilityTerm(LdotN, NdotV, roughness );//这里面做了1/4

				  //Normal  Distribution Function  类似于原来的BlinnPhong 高光项  ，但是这里能量守恒；越光滑 高光反射越集中在一点上且越亮
				  half D = NDFBlinnPhongNormalizedTerm(NdotH, PerceptualRoughnessToSpecPower(perceptualRoughness));

//#if UNITY_BRDF_GGX  --->>高配
//				  half V = SmithJointGGXVisibilityTerm(LdotN, NdotV, roughness);
//				  half D = GGXTerm(NdotH, roughness);
//#else 
//				  // Legacy  --->>低配
//				  half V = SmithBeckmannVisibilityTerm(LdotN, NdotV, roughness);
//				  half D = NDFBlinnPhongNormalizedTerm(NdotH, PerceptualRoughnessToSpecPower(perceptualRoughness));
//#endif

				  float3 brdfSpecular = _LightColor0 * fresnel * V * D *  UNITY_PI  ;//按道理应该是diffuse除以pi ，但是hack一下，仅对specualr * pi ，这样与传统的光照模型亮度做统一
				  brdfSpecular = max(0, brdfSpecular * LdotN);

				  float3 brdf = brdfSpecular + diffuse;

				  //IBL image-based lighting 
				  //大概做法是以烘焙好的贴图作为环境光的光照来源
				  //然后依据这个输入，再做一次类似于BRDF的计算（与直接光的brdf还是有些不同）
				  //ibl diffuse 的光 来源于 SH 重建的光照积分cubemap
				  //ibl specular 来源于 自己定义的cubemap 或者Unity 内置生成的 unity_SpecCube0 它是天空盒+probe 二者影响算出来的cubemap
				  float3 ibl;

				  //ibl diffuse  todo : 如果有lightmap 则直接读lightmap 否则自己算  LIGHTMAP_ON

				  //fresnelSchlickRoughness 
				  //基于经验的调整 roughness 参与运算得到的fresnel  , F90 取 1 - roughness
				  float3 fresnelRough = FresnelLerp(F0,saturate(1 - roughness), NdotV);
				  float kdIBL = (1 - fresnelRough) * (1 - metalic);

				  //重建Unity预处理生成的光照积分贴图，Unity把积分后的光照信息存储在了一组正交函数的系数上
				  //ShadeSH9函数可以重新取出了这部分的光照信息
				  //所谓的光照积分贴图是根据lightingSetting中的设置的环境光来源生成的,它可能是  SkyBox的CubeMap 、 梯度颜色 、 或者单纯就是 一个color
				  float3 ambient_contrib = ShadeSH9(float4(worldNormal, 1)); 
				  float3 ambient = 0.03 * albedo;
				  float3 iblDiffuse = max(half3(0, 0, 0), ambient + ambient_contrib);
				  float3 iblDiffuseResult = iblDiffuse * kdIBL * albedo;

				  //ibl specular

				  //环境光照贴图，根据粗糙度 做mipmap的采样
				  // 根据粗糙度得到CubeMap的LOD级别，越粗糙，越模糊
				  fixed cubeMip = perceptualRoughnessToMipmapLevel(perceptualRoughness);
				  //可选使用unity_SpecCube0或者自定义的cubemap
				  //如果有多个probe参与，还需要有unity_SpecCube0 、 unity_SpecCube1 等之间的混合，这里就不做了
				  //带LOD采CubeMap , 类似于 brdf specular 的 D 部分
				  fixed4 iblSpecular = texCUBElod(_CubeMap, fixed4(viewReflectDirection,cubeMip));
				  iblSpecular.rgb = DecodeHDR(iblSpecular, unity_SpecCube0_HDR);

				  //以下是Unity Standard BRDF 中的实现
				  float reflectivity = max(max(brdfSpecular.r, brdfSpecular.g), brdfSpecular.b);  // 反射率
				  // 这个是计算F90 , 这里的 F90 根据反射率去计算的 grazing其实就有这个中文意思
				  float grazingTerm = saturate(1 - roughness + reflectivity); //smooth + reflectivity

				  // 感觉类似于brdf的 V 部分 ，是一个近似拟合 ，Unreal和opengl 用的是一个 采LUT图的方案
				  fixed3 surfaceReduction = 1.0 / (roughness*roughness + 1.0); 
				  // FresnelLerp补上了最后一个Fresnel的影响
				  fixed3 iblSpecularResult = iblSpecular.rgb *  surfaceReduction  * FresnelLerp(_SpecColor, grazingTerm, NdotV);

				  ibl = iblDiffuseResult + iblSpecularResult;
				  

				  fixed3 finalColor =  brdf + ibl;

				  //ToneMapping
				  finalColor = ACESToneMapping(finalColor);

				  //debug code
				  //finalColor = brdfSpecular;// fixed3(NdotV, NdotV, NdotV);
				  fixed4 finalRGBA = fixed4(finalColor,mainColor.a)*_Color;
				  UNITY_APPLY_FOG(i.fogCoord, finalRGBA);
				  return finalRGBA;

				 }
				 ENDCG


			 }

		 }

		FallBack "Legacy Shaders/Diffuse"

}

